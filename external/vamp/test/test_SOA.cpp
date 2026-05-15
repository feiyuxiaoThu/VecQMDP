#include <iostream>
#include <string>

struct Agent
{
    std::string x, y, z;
};

struct World1
{
    Agent agents[100];
};

struct World2
{
    std::string xs[100];
    std::string ys[100];
    std::string zs[100];
};

int main()
{
    World1 world1;
    World2 world2;
}