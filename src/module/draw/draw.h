//
// Created by Dongmin on 25. 4. 28.
//

#ifndef DRAW_H
#define DRAW_H

#include "module/module.h"
#include "utils/ColorMap/ColorMap.h"

class Draw : public Module {
public:
    Draw() {};
    ~Draw() {};
    bool Initialize() override;
    void Release() override;
    void Run() override;
private:
    int height_;
    int width_;
    int seg_width_;
    int seg_height_;
    ColorMap colormap_;
};



#endif //DRAW_H
