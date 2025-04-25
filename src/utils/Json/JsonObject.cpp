#include "JsonObject.h"
#include "../FileSystem/FileSystem.h"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace nlohmann;

namespace custom
{
    void to_json(nlohmann::json& j, const JsonVec2& p)
    {
        j = { { "x", p.x }, { "y", p.y } };
    }
    void from_json(const nlohmann::json& j, JsonVec2& p)
    {
        p.x = j.at("x").get<float>();
        p.y = j.at("y").get<float>();
    }
}

void JsonObject::copy_from(JsonObject& target)
{
    j = target.j;
}

void JsonObject::clear()
{
    j.clear();
}

void JsonObject::write(string layered_key, vector<char>& value)
{
    vector<std::string> splits = FileSystem::split(layered_key, "/");
    vector<ordered_json> buf;
    int i = (int)splits.size() - 1;
    int buf_i = 0;
    while (true)
    {
        ordered_json tmp;
        buf.push_back(tmp);
        std::string key = splits[i];
        if (buf_i == 0)
        {
            buf[buf_i][key] = value;
        }
        else
        {
            buf[buf_i][key].merge_patch(buf[buf_i - 1]);
        }
        i--;
        buf_i++;
        if (i < 0) break;
    }
    j.merge_patch(buf[splits.size() - 1]);
}

void JsonObject::write(string layered_key, vector<float>& value)
{
    vector<std::string> splits = FileSystem::split(layered_key, "/");
    vector<ordered_json> buf;
    int i = (int)splits.size() - 1;
    int buf_i = 0;
    while (true)
    {
        ordered_json tmp;
        buf.push_back(tmp);
        std::string key = splits[i];
        if (buf_i == 0)
        {
            buf[buf_i][key] = value;
        }
        else
        {
            buf[buf_i][key].merge_patch(buf[buf_i - 1]);
        }
        i--;
        buf_i++;
        if (i < 0) break;
    }
    j.merge_patch(buf[splits.size() - 1]);
}

void JsonObject::write(string layered_key, vector<int>& value)
{
    vector<std::string> splits = FileSystem::split(layered_key, "/");
    vector<ordered_json> buf;
    int i = (int)splits.size() - 1;
    int buf_i = 0;
    while (true)
    {
        ordered_json tmp;
        buf.push_back(tmp);
        std::string key = splits[i];
        if (buf_i == 0)
        {
            buf[buf_i][key] = value;
        }
        else
        {
            buf[buf_i][key].merge_patch(buf[buf_i - 1]);
        }
        i--;
        buf_i++;
        if (i < 0) break;
    }
    j.merge_patch(buf[splits.size() - 1]);
}

void JsonObject::write(string layered_key, vector<custom::JsonVec2>& value)
{
    vector<std::string> splits = FileSystem::split(layered_key, "/");
    vector<ordered_json> buf;
    int i = (int)splits.size() - 1;
    int buf_i = 0;
    while (true)
    {
        ordered_json tmp;
        buf.push_back(tmp);
        std::string key = splits[i];
        if (buf_i == 0)
        {
            json j = value;
            buf[buf_i][key] = j;
            //buf[buf_i][key] = value;
        }
        else
        {
            buf[buf_i][key].merge_patch(buf[buf_i - 1]);
        }
        i--;
        buf_i++;
        if (i < 0) break;
    }
    j.merge_patch(buf[splits.size() - 1]);
}

void JsonObject::write(std::string layered_key, int value)
{
    vector<std::string> splits = FileSystem::split(layered_key, "/");
    vector<ordered_json> buf;
    int i = (int)splits.size() - 1;
    int buf_i = 0;
    while (true)
    {
        ordered_json tmp;
        buf.push_back(tmp);
        std::string key = splits[i];
        if (buf_i == 0)
        {
            buf[buf_i][key] = value;
        }
        else
        {
            buf[buf_i][key].merge_patch(buf[buf_i - 1]);
        }
        i--;
        buf_i++;
        if (i < 0) break;
    }
    j.merge_patch(buf[splits.size() - 1]);
}

void JsonObject::write(string layered_key, bool value)
{
    write(layered_key, (int)value);
}

void JsonObject::write(string layered_key, float value)
{
    vector<std::string> splits = FileSystem::split(layered_key, "/");
    vector<ordered_json> buf;
    int i = (int)splits.size() - 1;
    int buf_i = 0;
    while (true)
    {
        ordered_json tmp;
        buf.push_back(tmp);
        std::string key = splits[i];
        if (buf_i == 0)
        {
            buf[buf_i][key] = value;
        }
        else
        {
            buf[buf_i][key].merge_patch(buf[buf_i - 1]);
        }
        i--;
        buf_i++;
        if (i < 0) break;
    }
    j.merge_patch(buf[splits.size() - 1]);
}

void JsonObject::write(string layered_key, double value)
{
    vector<std::string> splits = FileSystem::split(layered_key, "/");
    vector<ordered_json> buf;
    int i = (int)splits.size() - 1;
    int buf_i = 0;
    while (true)
    {
        ordered_json tmp;
        buf.push_back(tmp);
        std::string key = splits[i];
        if (buf_i == 0)
        {
            buf[buf_i][key] = value;
        }
        else
        {
            buf[buf_i][key].merge_patch(buf[buf_i - 1]);
        }
        i--;
        buf_i++;
        if (i < 0) break;
    }
    j.merge_patch(buf[splits.size() - 1]);
}

void JsonObject::write(string layered_key, string value)
{
    vector<std::string> splits = FileSystem::split(layered_key, "/");
    vector<ordered_json> buf;
    int i = (int)splits.size() - 1;
    int buf_i = 0;
    while (true)
    {
        ordered_json tmp;
        buf.push_back(tmp);
        std::string key = splits[i];
        if (buf_i == 0)
        {
            buf[buf_i][key] = value;
        }
        else
        {
            buf[buf_i][key].merge_patch(buf[buf_i - 1]);
        }
        i--;
        buf_i++;
        if (i < 0) break;
    }
    j.merge_patch(buf[splits.size() - 1]);
}

bool JsonObject::read(string layered_key, int& value, int default_value)
{
    vector<string> splits = FileSystem::split(layered_key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            value = default_value;
            return false;
        }
    }
    flag = it.contains(splits[splits.size() - 1]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    value = it[splits[splits.size() - 1]];
    return true;
}

bool JsonObject::read(string layered_key, bool& value, bool default_value)
{
    vector<string> splits = FileSystem::split(layered_key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            value = default_value;
            return false;
        }
    }
    flag = it.contains(splits[splits.size() - 1]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    value = it[splits[splits.size() - 1]];
    return true;
}

bool JsonObject::read(string layered_key, float& value, float default_value)
{
    vector<string> splits = FileSystem::split(layered_key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            value = default_value;
            return false;
        }
    }
    flag = it.contains(splits[splits.size() - 1]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    value = it[splits[splits.size() - 1]];
    return true;
}

bool JsonObject::read(string layered_key, double& value, double default_value)
{
    vector<string> splits = FileSystem::split(layered_key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            value = default_value;
            return false;
        }
    }
    flag = it.contains(splits[splits.size() - 1]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    value = it[splits[splits.size() - 1]];
    return true;
}

bool JsonObject::read(string layered_key, string& value, string default_value)
{
    vector<string> splits = FileSystem::split(layered_key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            value = default_value;
            return false;
        }
    }
    flag = it.contains(splits[splits.size() - 1]);
    if (flag == false)
    {
        value = default_value;
        return false;
    }
    value = it[splits[splits.size() - 1]];
    return true;
}

bool JsonObject::read(string layered_key, vector<char>& value)
{
    vector<string> splits = FileSystem::split(layered_key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        return false;
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            return false;
        }
    }
    flag = it.contains(splits[splits.size() - 1]);
    if (flag == false)
    {
        return false;
    }
    value = it[splits[splits.size() - 1]];
    return true;
}

bool JsonObject::read(string layered_key, vector<float>& value)
{
    vector<string> splits = FileSystem::split(layered_key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        return false;
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            return false;
        }
    }
    flag = it.contains(splits[splits.size() - 1]);
    if (flag == false)
    {
        return false;
    }
    value.assign(it[splits[splits.size() - 1]].begin(), it[splits[splits.size() - 1]].end());
    return true;
}

bool JsonObject::read(string layered_key, vector<custom::JsonVec2>& value)
{
    vector<string> splits = FileSystem::split(layered_key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        return false;
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            return false;
        }
    }
    flag = it.contains(splits[splits.size() - 1]);
    if (flag == false)
    {
        return false;
    }
    value.assign(it[splits[splits.size() - 1]].begin(), it[splits[splits.size() - 1]].end());
    return true;
}

int JsonObject::get_element_count(string key)
{
    vector<string> splits = FileSystem::split(key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        return false;
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            return false;
        }
    }
    return it.size();
}

JsonObject JsonObject::get_element(string key, int index)
{
    vector<string> splits = FileSystem::split(key, "/");
    bool flag = j.contains(splits[0]);
    if (flag == false)
    {
        return JsonObject();
    }
    auto it = j[splits[0]];
    for (int i = 1; i < splits.size() - 1; i++)
    {
        string key = splits[i];
        bool flag = it.contains(key);
        if (flag == true)
        {
            it = it[key];
        }
        else
        {
            return JsonObject();
        }
    }
    JsonObject tmp;
    auto itr = it.items().begin();
    for (int i = 0; i < index; i++)
    {
        itr++;
    }
    tmp.j = *itr;
    return tmp;
}

string JsonObject::get_element_key()
{
    string value = j.items().begin().key();
    return value;
}

void JsonObject::save(string path)
{
    std::ofstream o(path);
    o << std::setw(4) << j << std::endl;
}

void JsonObject::load(string path)
{
    if (FileSystem::exist(path) == false)
    {
        printf("Cannot Find Json(%s)\n", path.c_str());
        return;
    }
    j.clear();
    std::ifstream i(path);
    i >> j;
}

void JsonObject::load_from_string(string str)
{
    j.clear();
    j = j.parse(str);
}

string JsonObject::save_to_string(int indent)
{
    return j.dump(indent);
}

void JsonObject::set_string(string key, string value)
{
    write(key, value);
}

void JsonObject::set_string(string key, const char* value)
{
    write(key, value);
}

void JsonObject::set_int(string key, int value)
{
    write(key, value);
}

void JsonObject::set_float(string key, float value)
{
    write(key, value);
}

void JsonObject::set_array(string key, vector<int>& value)
{
    write(key, value);
}

string JsonObject::get_string(string key)
{
    string tmp = "";
    read(key, tmp);
    return tmp;
}

int JsonObject::get_int(string key)
{
    int tmp = 0;
    read(key, tmp);
    return tmp;
}

float JsonObject::get_float(string key)
{
    float tmp = 0;
    read(key, tmp);
    return tmp;
}

vector<float> JsonObject::get_array(string key)
{
    vector<float> tmp;
    read(key, tmp);
    return tmp;
}