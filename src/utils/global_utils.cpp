/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include "utils/global_utils.hpp"
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <queue>

namespace vec_qmdp
{
    namespace utils
    {
        // Implementation of logging system
        namespace
        {
            std::ofstream log_file_stream;
            LogLevel      current_log_level = LogLevel::INFO;
            std::mutex    log_mutex;
            bool          logging_initialized = false;

            // Get string representation of current timestamp
            std::string getCurrentTimeStamp()
            {
                auto now = std::chrono::system_clock::now();
                auto time = std::chrono::system_clock::to_time_t(now);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

                std::stringstream ss;
                ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
                ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
                return ss.str();
            }

            // Convert LogLevel to string
            std::string logLevelToString(LogLevel level)
            {
                switch (level)
                {
                case LogLevel::DEBUG:
                    return "[DEBUG]";
                case LogLevel::INFO:
                    return "[INFO]";
                case LogLevel::WARNING:
                    return "[WARNING]";
                case LogLevel::ERROR:
                    return "[ERROR]";
                default:
                    return "[UNKNOWN]";
                }
            }
        } // namespace

        class ThreadPool::Impl
        {
          public:
            Impl(size_t num_threads) : stop(false)
            {
                for (size_t i = 0; i < num_threads; ++i)
                {
                    workers.emplace_back(
                        [this]
                        {
                            while (true)
                            {
                                std::function<void()> task;
                                {
                                    std::unique_lock<std::mutex> lock(queue_mutex);
                                    condition.wait(lock, [this] { return stop || !tasks.empty(); });
                                    if (stop && tasks.empty())
                                        return;
                                    task = std::move(tasks.front());
                                    tasks.pop();
                                }
                                task();
                            }
                        });
                }
            }

            ~Impl()
            {
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    stop = true;
                }
                condition.notify_all();
                for (std::thread &worker : workers)
                    worker.join();
            }

            void addTask(std::function<void()> task)
            {
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);

                    if (stop)
                        throw std::runtime_error("enqueue on stopped ThreadPool");

                    tasks.push(std::move(task));
                }
                condition.notify_one();
            }

          public:
            std::vector<std::thread>          workers;
            std::queue<std::function<void()>> tasks;
            std::mutex                        queue_mutex;
            std::condition_variable           condition;
            bool                              stop;
        };

        ThreadPool::ThreadPool(size_t num_threads) : pImpl(new Impl(num_threads)) {}
        ThreadPool::~ThreadPool() = default;

        void ThreadPool::enqueueTask(std::function<void()> task) { pImpl->addTask(std::move(task)); }

        void enableSIMD() {}

        bool isSIMDAvailable() { return true; }

        int getSIMDWidth() { return 8; }

        // 1. Global aggregated data (all thread data is merged here)
        std::map<std::string, PerformanceData> g_global_perf_data;
        std::mutex                             g_perf_mutex; // Mutex protecting global data

        // 2. Thread-local data (exclusive per thread, no locking required, extremely fast)
        thread_local std::map<std::string, PerformanceData> t_local_perf_data;

        // Recursively merge two PerformanceData trees
        void mergePerfDataRecursive(PerformanceData &dest, const PerformanceData &src)
        {
            dest.total_time_ms += src.total_time_ms;
            dest.call_count += src.call_count;

            for (const auto &[name, src_sub_ptr] : src.sub_functions)
            {
                if (dest.sub_functions.find(name) == dest.sub_functions.end())
                {
                    // If dest does not have this subtree, create a new entry
                    dest.sub_functions[name] = std::make_shared<PerformanceData>();
                }
                // Recursively merge subtrees
                mergePerfDataRecursive(*dest.sub_functions[name], *src_sub_ptr);
            }
        }

        // [IMPORTANT] Must be called before each thread exits!
        void commitPerformanceData()
        {
            // If the current thread has no data, return immediately
            if (t_local_perf_data.empty())
                return;

            std::lock_guard<std::mutex> lock(g_perf_mutex);

            // Merge local map into global map
            for (const auto &[name, local_data] : t_local_perf_data)
            {
                // Merge top-level node and its subtree
                mergePerfDataRecursive(g_global_perf_data[name], local_data);
            }

            // Clear local data to prevent duplicate commits
            t_local_perf_data.clear();
        }

        // ---------------- Print Logic ----------------

        void printSubFunctionData(const std::string &func_name, const std::shared_ptr<PerformanceData> &data,
                                  int indent)
        {
            if (!data)
                return;

            double avg_time = data->call_count > 0 ? data->total_time_ms / data->call_count : 0;

            std::string prefix(indent * 2, ' ');
            if (indent > 0)
                prefix += "- ";

            std::cout << prefix << func_name << ": time: " << std::fixed << std::setprecision(3) << data->total_time_ms
                      << "ms"
                      << ", calls: " << data->call_count << ", avg: " << avg_time << "ms" << std::endl;

            for (const auto &sub_entry : data->sub_functions)
            {
                printSubFunctionData(sub_entry.first, sub_entry.second, indent + 1);
            }
        }

        void printPerformanceData()
        {
            // Before printing, ensure all worker threads have committed or joined
            // This prints the global g_global_perf_data
            commitPerformanceData();

            std::lock_guard<std::mutex> lock(g_perf_mutex); // Lock during print to guard against concurrent writes

            std::cout << "\n================ Performance Profiling ================\n";
            for (const auto &[name, data] : g_global_perf_data)
            {
                // Reuse printSubFunctionData logic for formatting, or print manually inline
                double avg_time = data.call_count > 0 ? data.total_time_ms / data.call_count : 0;

                std::cout << name << ": time: " << std::fixed << std::setprecision(3) << data.total_time_ms << "ms"
                          << ", calls: " << data.call_count << ", avg: " << avg_time << "ms" << std::endl;

                for (const auto &sub_entry : data.sub_functions)
                {
                    printSubFunctionData(sub_entry.first, sub_entry.second, 1);
                }
            }
            std::cout << "=======================================================\n";
        }

        void initializeLogger(const std::string &log_file, LogLevel level)
        {
            {
                std::lock_guard<std::mutex> lock(log_mutex);

                // Close previous log file if open
                if (log_file_stream.is_open())
                {
                    log_file_stream.close();
                }

                // Open new log file in truncation mode to clear existing content
                if (!log_file.empty())
                {
                    log_file_stream.open(log_file, std::ios::out | std::ios::trunc);
                    if (!log_file_stream.is_open())
                    {
                        std::cerr << "Failed to open log file: " << log_file << std::endl;
                    }
                }

                current_log_level = level;
                logging_initialized = true;
            }

            // Log initialization message
            log(LogLevel::INFO, "Logger initialized with level: " + logLevelToString(level));
        }

        void log(LogLevel level, const std::string &message, bool append_newline)
        {
            // Ignore messages below the current log level threshold
            if (static_cast<int>(level) < static_cast<int>(current_log_level))
            {
                return;
            }

            std::lock_guard<std::mutex> lock(log_mutex);

            // Build log message
            std::string timestamp = getCurrentTimeStamp();
            std::string level_str = logLevelToString(level);
            std::string formatted_message = timestamp + " " + level_str + " " + message;

            // Write to log file if open
            if (log_file_stream.is_open())
            {
                log_file_stream << formatted_message << std::endl;
                log_file_stream.flush();
            }

            // Output to console
            if (append_newline)
            {
                if (level == LogLevel::ERROR)
                {
                    std::cerr << formatted_message << std::endl;
                }
                else
                {
                    std::cout << formatted_message << std::endl;
                }
            }
            else
            {
                if (level == LogLevel::ERROR)
                {
                    std::cerr << formatted_message;
                }
                else
                {
                    std::cout << formatted_message;
                }
            }
        }

        PerformanceMetrics getPerformanceMetrics() { return PerformanceMetrics(); }

        void resetPerformanceMetrics() {}

    } // namespace utils
} // namespace vec_qmdp