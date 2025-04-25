//
// Created by Dongmin on 25. 4. 24.
//

#ifndef DISPLAY_H
#define DISPLAY_H

#include <SDL3/SDL_render.h>

#include "module/module.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#include <GLFW/glfw3.h>


class Display : public Module {
public:
    Display() {};
    ~Display() {};
    bool Initialize() override;
    void Release() override;
    void Run() override;

private:
    GLuint frame_texture_id_;
};



#endif //DISPLAY_H
