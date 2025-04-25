
// Main code
#include "application/application.h"
#include "src/imgui_trt.h"

int main(int, char**)
{
    ImGuiTRT trt;
    if (!trt.Initialize())
        return 0;
    trt.Run();
    trt.Release();
    return 0;
}