//
// Created by gopizza on 25. 4. 24.
//

#include "application.h"



bool Application::Initialize() {
    // Create window with graphics context
    if (!glfwInit())
        return false;
    window_ = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window_ == nullptr)
        return false;
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init();
}

void Application::Release() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void Application::Run() {
    // Main loop
    uint index;
    if (data_store_->module_index_queues_[input_module_index_].try_pop(index)) {
        // Start the ImGui frame
        // ImGui_ImplOpenGL3_NewFrame();
        // ImGui_ImplGlfw_NewFrame();
        // ImGui::NewFrame();
        //
        // // Render your GUI here
        //
        // // Rendering
        // ImGui::Render();
        // int display_w, display_h;
        // glfwGetFramebufferSize(window_, &display_w, &display_h);
        // glViewport(0, 0, display_w, display_h);
        // glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        // glClear(GL_COLOR_BUFFER_BIT);
        // ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        //
        // glfwSwapBuffers(window_);
        stop_flag_ = glfwWindowShouldClose(window_);
        data_store_->module_index_queues_[output_module_index_].push(index);

    }
}