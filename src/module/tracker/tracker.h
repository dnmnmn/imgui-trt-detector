//
// Created by Dongmin on 25. 3. 17.
//

#ifndef TRACKER_H
#define TRACKER_H

#include "module/module.h"
#include "hungarian/hungarian_algorithm.hpp"
#include "data/box/trackbox.h"

class Tracker : public Module {
public:
    Tracker() {};
    ~Tracker() {};

    bool Initialize() override;
    void Release() override;
    void Run() override;

    void PopBox();
    void Match(vector<Bbox> &_objects);
    void Update(vector<TrackBox> &_objects);
    void Predict();
protected:
    void MakeNewTrackBox(Bbox &_obj);
    void CorrectBox(Bbox &_obj, TrackBox &_track);
    void GetCostMatrix(vector<Bbox> &_objects, vector<vector<double>> &_cost, vector<int> &_track_idx_vec);
public:
    std::shared_ptr<HungarianAlgorithm> matcher_;
    std::vector<TrackBox> track_boxes_vector_;
    std::list<int> track_alive_boxes_list_;
    std::queue<int> track_recycle_queue_;

    int width_, height_;
    int max_frame_;
    int frame_idx_;
    int track_id_ = 0;
};

#endif //TRACKER_H
