//
// Created by Dongmin on 25. 3. 17.
//

#include "tracker.h"

bool Tracker::Initialize() {
    name_ = "Tracker";
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    JsonObject config_json;
    config_json.load(config_path_);
    debug_time_ = config_json.get_int("Engine/Log/DebugTime") * 1000;
    width_ = config_json.get_int("Engine/Segment/Width");
    height_ = config_json.get_int("Engine/Segment/Height");
    max_frame_ = config_json.get_int("Engine/Filter/MaxFrame");
    matcher_ = std::make_shared<HungarianAlgorithm>();
    track_alive_boxes_list_.clear();
    track_boxes_vector_.resize(200);
    for (int i = 0; i < 200; i++) {
        track_recycle_queue_.push(i);
    }

    frame_idx_ = 0;
    return true;
}

void Tracker::Release() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    track_boxes_vector_.clear();
}

void Tracker::Run() {
    uint index = 0;
    if(data_store_->module_index_queues_[input_module_index_].try_pop(index)) {
        auto container = data_store_->contaiers_[index];

        if (!track_alive_boxes_list_.empty()) {
            auto index = track_alive_boxes_list_.begin();
            data_store_->tracker_box_mutex_.lock();
            data_store_->tracker_box_.Copy(track_boxes_vector_[*index]);
            data_store_->tracker_box_.active_ = true;
            data_store_->tracker_box_mutex_.unlock();
        }

        if (container->org_image_->empty()) {
            data_store_->module_index_queues_[output_module_index_].push(index);
            return;
        }
        auto det_objs = data_store_->contaiers_[index]->bboxes_;
        Match(det_objs);
        Update(data_store_->contaiers_[index]->track_boxes_);
        // Predict();

        ++pass_count_;
        data_store_->module_index_queues_[output_module_index_].push(index);
    }
    else {
        ++fail_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }
}

void Tracker::Match(std::vector<Bbox> &_objects) {
    if (_objects.empty()) {
        return;
    }
    if(track_alive_boxes_list_.empty()) {
        for(auto &obj : _objects) {
            MakeNewTrackBox(obj);
        }
        return;
    }

    // Make cost(iou) matrix
    std::vector<int> track_idx_vec(track_alive_boxes_list_.size(),0);
    std::vector<std::vector<double>> cost(_objects.size(), std::vector<double>(track_idx_vec.size(), 0));
    GetCostMatrix(_objects, cost, track_idx_vec);

    std::vector<int> assignment;
    auto results = matcher_->Solve(cost, assignment);
    for(int i = 0; i < assignment.size(); i++) {
        if(assignment[i] == -1) {
            MakeNewTrackBox(_objects[i]);
        }
        else if (cost[i][assignment[i]] < 0.50f) {
            CorrectBox(_objects[i], track_boxes_vector_[track_idx_vec[assignment[i]]]);
        }
        else if (cost[i][assignment[i]] >= 0.80f) {
            MakeNewTrackBox(_objects[i]);
        }
    }
}

void Tracker::Update(vector<TrackBox> &_objects) {
    _objects.clear();
    auto track_idx = track_alive_boxes_list_.begin();
    while(track_idx != track_alive_boxes_list_.end()) {
        uint8_t index = *track_idx;
        bool is_alive = track_boxes_vector_[index].Update();
        if(!is_alive) {
            track_recycle_queue_.push(index);
            track_idx = track_alive_boxes_list_.erase(track_idx);
        }
        else {
            _objects.push_back(track_boxes_vector_[index]);
            ++track_idx;
        }
    }
}

void Tracker::Predict() {
    auto track_idx = track_alive_boxes_list_.begin();
    while(track_idx != track_alive_boxes_list_.end()) {
        uint8_t index = *track_idx++;
        if ((track_boxes_vector_)[index].is_matched_ == false) {
            continue;
        }
        (track_boxes_vector_)[index].Predict();
        (track_boxes_vector_)[index].is_matched_ = true;
    }
}

void Tracker::MakeNewTrackBox(Bbox &_det_box) {
    if (_det_box.score_ < 0.7f) {
        return;
    }
    if (track_recycle_queue_.empty() == false) {
        int index = track_recycle_queue_.front();
        track_recycle_queue_.pop();

        track_alive_boxes_list_.push_back(index);
        track_boxes_vector_[index].Copy(_det_box);
        track_boxes_vector_[index].track_id_ = track_id_++;
        track_alive_boxes_list_.push_back(index);
        // std::cout << "MakeNewTrackBox: " << index << std::endl;
    }
    else {
        std::cout << "Error: MakeNewTrackBox" << std::endl;
    }
}

void Tracker::CorrectBox(Bbox &_det_box, TrackBox &_track_box) {
    _track_box.Correct(_det_box);
}

void Tracker::GetCostMatrix(vector<Bbox> &_objects, vector<vector<double> > &_cost, vector<int> &_track_idx_vec) {
    auto track_idx = track_alive_boxes_list_.begin();
    for(int i = 0 ; i < track_alive_boxes_list_.size(); i++) {
        _track_idx_vec[i] = (*track_idx++);
    }

    for(int i = 0;i < _objects.size(); i++)
    {
        auto det_obj = _objects[i];

        for(int j = 0; j < _track_idx_vec.size(); j++)
        {
            auto track_obj = (track_boxes_vector_)[_track_idx_vec[j]];
            // 1. class Similarity
            // if (track_obj.second.class_id_ != det_obj.class_id_) {
            //     _cost[i][j] = 1.0f;
            //     continue;
            // }
            // 2. IoU
            _cost[i][j] = 1 - track_obj.IoU(det_obj);
            // 3. Distance Weight
            //_cost[i][j] += track_obj.second.Distance(det_obj);
            //
            _cost[i][j] = std::clamp(_cost[i][j], 0.0, 1.0);
            //
            // if (cost[i][j] > 0.6) cost[i][j] = 1;

        }
    }
}
