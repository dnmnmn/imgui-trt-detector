//
// Created by Dongmin on 25. 4. 24.
//

#ifndef DISPLAY_H
#define DISPLAY_H

#include "module/module.h"
#include "imgui.h"

#include <GLFW/glfw3.h>


class Display : public Module {
public:
    Display() {};
    ~Display() {};
    bool Initialize() override;
    void Release() override;
    void Run() override;

    void RunThread() override;
private:
    GLuint frame_texture_id_;
    GLFWwindow* window_;
};



#endif //DISPLAY_H
