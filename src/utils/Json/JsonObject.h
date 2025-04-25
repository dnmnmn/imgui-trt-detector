#pragma once

#include "Json.hpp"
#include <iostream>
using namespace std;

namespace custom
{
    struct JsonVec2
    {
        float x, y;
    };
}

namespace custom
{
    void to_json(nlohmann::json& j, const JsonVec2& p);
    void from_json(const nlohmann::json& j, JsonVec2& p);
}

class JsonObject
{
private:
    nlohmann::ordered_json j;
public:
    void copy_from(JsonObject& target);
    void clear();
public:
    void write(string layered_key, int value);
    void write(string layered_key, bool value);
    void write(string layered_key, float value);
    void write(string layered_key, double value);
    void write(string layered_key, string value);
    void write(string layered_key, vector<char>& value);
    void write(string layered_key, vector<float>& value);
    void write(string layered_key, vector<int>& value);
    void write(string layered_key, vector<custom::JsonVec2>& value);
public:
    bool read(string layered_key, int& value, int default_value = 0);
    bool read(string layered_key, bool& value, bool default_value = false);
    bool read(string layered_key, float& value, float default_value = 0.0f);
    bool read(string layered_key, double& value, double default_value = 0.0f);
    bool read(string layered_key, string& value, string default_value = "");
    bool read(string layered_key, vector<char>& value);
    bool read(string layered_key, vector<float>& value);
    bool read(string layered_key, vector<custom::JsonVec2>& value);
public:
    void set_string(string key, string value);
    void set_string(string key, const char* value);
    void set_int(string key, int value);
    void set_float(string key, float value);
    void set_array(string key, vector<int>& value);
public:
    string get_string(string key);
    int get_int(string key);
    float get_float(string key);
    vector<float> get_array(string key);
    int get_element_count(string key);
    JsonObject get_element(string key, int index);
    string get_element_key();
public:
    void save(string path);
    string save_to_string(int indent = -1);
    void load(string path);
    void load_from_string(string str);
};
