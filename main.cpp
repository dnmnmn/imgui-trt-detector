
// Main code
#include "src/imgui_trt.h"

// static void glfw_error_callback(int error, const char* description)
// {
//     fprintf(stderr, "GLFW Error %d: %s\n", error, description);
// }

int main(int, char**)
{
    // glfwSetErrorCallback(glfw_error_callback);
    ImGuiTRT trt;
    if (!trt.Initialize())
        return 0;
    trt.Run();
    trt.Release();
    return 0;
}