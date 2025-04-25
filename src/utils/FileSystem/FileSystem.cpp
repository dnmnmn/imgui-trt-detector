#if defined(_MSC_VER)
#include <windows.h>
#include <filesystem>
namespace fs = std::filesystem;
#elif defined(__APPLE__)
#include <limits.h>
#include <unistd.h>
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <filesystem>
#include <limits.h>
#include <unistd.h>

namespace fs = std::filesystem;
#endif
#include <fstream>

#if defined(USE_NFD)
#include <nfd.h>
#endif
#include "FileSystem.h"

std::string FileSystem::last_dialog_path = "";

std::string FileSystem::get_app_path()
{
#if defined(_MSC_VER)
    char result[512]{ 0, };
    std::string str = std::string(result, GetModuleFileNameA(NULL, result, 512));
    return str.substr(0, str.find_last_of("\\"));
#else
    char result[PATH_MAX];
    char* res = realpath(".", result);
    if (res)
    {
        string tmp = result;
        return tmp;
    }
    else
    {
        return "";
    }
    //ssize_t count = readlink("proc/self/exe", result, 512);
    //std::string str = std::string(result, (count > 0) ? count : 0);
    //return str.substr(0, str.find_last_of("/"));
#endif
}

string FileSystem::get_dialog_last_path()
{
    return last_dialog_path;
}

bool FileSystem::open_file_dialog(std::string filter, std::string default_path, std::string* result)
{
#if defined(USE_NFD)
    char* out_path = NULL;
    char* filter_str = (char*)filter.c_str();
    char* default_path_str = (char*)default_path.c_str();
    nfdresult_t res = NFD_OpenDialog(filter_str, default_path_str, &out_path);
    if (res == NFD_OKAY)
    {
        last_dialog_path = out_path;
        *result = out_path;
        return true;
    }
    else if (res == NFD_CANCEL)
    {
        //cancel
    }
    else
    {
        //error
    }
#endif
    * result = "";
    return false;
}

bool FileSystem::open_multiple_file_dialog(std::string filter, std::string default_path, vector<std::string>* result)
{
    result->clear();
#if defined(USE_NFD)
    char* out_path = NULL;
    char* filter_str = (char*)filter.c_str();
    char* default_path_str = (char*)default_path.c_str();
    nfdpathset_t pathset;
    nfdresult_t res = NFD_OpenDialogMultiple(filter_str, default_path_str, &pathset);
    if (res == NFD_OKAY)
    {
        for (int i = 0; i < pathset.count; i++)
        {
            out_path = (char*)pathset.buf + pathset.indices[i];
            result->push_back(out_path);
            last_dialog_path = out_path;
        }
        return true;
    }
    else if (res == NFD_CANCEL)
    {
        //cancel
    }
    else
    {
        //error
    }
#endif
    return false;
}

bool FileSystem::save_file_dialog(std::string filter, std::string default_path, std::string* result)
{
#if defined(USE_NFD)
    char* out_path = NULL;
    char* filter_str = (char*)filter.c_str();
    char* default_path_str = (char*)default_path.c_str();
    nfdresult_t res = NFD_SaveDialog(filter_str, default_path_str, &out_path);
    if (res == NFD_OKAY)
    {
        last_dialog_path = out_path;
        *result = out_path;
        return true;
    }
    else if (res == NFD_CANCEL)
    {
        //cancel
    }
    else
    {
        //error
    }
#endif
    * result = "";
    return false;
}

bool FileSystem::open_folder_dialog(std::string default_path, std::string* result)
{
#if defined(USE_NFD)
    char* out_path = NULL;
    char* default_path_str = (char*)default_path.c_str();
    nfdresult_t res = NFD_PickFolder(default_path_str, &out_path);
    if (res == NFD_OKAY)
    {
        *result = out_path;
        return true;
    }
    else if (res == NFD_CANCEL)
    {
        //cancel
    }
    else
    {
        //error
    }
#endif
    * result = "";
    return false;
}

bool FileSystem::save_folder_dialog(std::string default_path, std::string* result)
{
#if defined(USE_NFD)
    char* out_path = NULL;
    char* default_path_str = (char*)default_path.c_str();
    nfdresult_t res = NFD_PickFolder(default_path_str, &out_path);
    if (res == NFD_OKAY)
    {
        *result = out_path;
        return true;
    }
    else if (res == NFD_CANCEL)
    {
        //cancel
    }
    else
    {
        //error
    }
#endif
    * result = "";
    return false;
}

std::vector<string> FileSystem::get_directories(string dir_path)
{
    vector<string> result;
    for (const auto& entry : fs::directory_iterator(dir_path))
    {
        if (entry.status().type() == fs::file_type::directory) result.push_back(entry.path().string());
    }
    return result;
}

std::vector<string> FileSystem::get_files(string dir_path, string filter)
{
    vector<string> result_vec;
    vector<string> tmp = split(filter, ".");
    string split_filter = tmp[tmp.size() - 1];
    for (const auto& entry : fs::directory_iterator(dir_path))
    {
        if (entry.status().type() != fs::file_type::directory)
        {
            vector<string> splits = split(entry.path().string(), ".");
            if (splits[splits.size() - 1] == split_filter) result_vec.push_back(entry.path().string());
        }
        else if (entry.status().type() == fs::file_type::directory)
        {
            vector<string> buf = get_files(entry.path().string(), filter);
            for (int i = 0; i < buf.size(); i++)
            {
                vector<string> splits = split(buf[i], ".");
                if (splits[splits.size() - 1] == split_filter) result_vec.push_back(buf[i]);
            }
        }
    }
    return result_vec;
}

vector<string> FileSystem::split(string stringToBeSplitted, string delimeter)
{
    std::vector<std::string> splittedString;
    int startIndex = 0;
    int endIndex = 0;
    while ((endIndex = (int)stringToBeSplitted.find(delimeter, startIndex)) < stringToBeSplitted.size())
    {

        std::string val = stringToBeSplitted.substr(startIndex, endIndex - startIndex);
        splittedString.push_back(val);
        startIndex = endIndex + (int)delimeter.size();

    }
    if (startIndex < stringToBeSplitted.size())
    {
        std::string val = stringToBeSplitted.substr(startIndex);
        splittedString.push_back(val);
    }
    return splittedString;
}

vector<string> FileSystem::split(string stringToBeSplitted, vector<string> delimeters, vector<string> remove)
{
    vector<string> buffer, splittedString, buffer2;
    splittedString.push_back(stringToBeSplitted);

    for (int i = 0; i < delimeters.size(); i++)
    {
        buffer2.clear();
        for (int j = 0; j < splittedString.size(); j++)
        {
            buffer = split(splittedString[j], delimeters[i]);
            for (int k = 0; k < buffer.size(); k++)
            {
                buffer2.push_back(buffer[k]);
            }
        }
        splittedString.clear();
        for (int j = 0; j < buffer2.size(); j++)
        {
            bool flag = false;
            for (int k = 0; k < remove.size(); k++)
            {
                if (buffer2[j] == remove[k])
                {
                    flag = true;
                    break;
                }
            }
            if (flag == false) splittedString.push_back(buffer2[j]);
        }
    }
    return splittedString;
}

vector<string> FileSystem::read_all_textline(string path)
{
    vector<string> vec;
#if defined(_MSC_VER)
    ifstream fsa(path);
    string line;
    while (getline(fsa, line))
    {
        vec.push_back(line);
    }
    fsa.close();
#else
    ifstream fsa(path);
    string line;
    while (getline(fsa, line))
    {
        vec.push_back(line);
    }
    fsa.close();
#endif
    return vec;
}

void FileSystem::write_all_text(string path, string contents)
{
    //#ifdef _WIN32
    ofstream fsa(path, ios::app);
    fsa << contents;
    fsa.close();
    //#endif
}

void FileSystem::write_add_text(string path, string contents)
{
    //#ifdef _WIN32
    ofstream fsa(path, ios::app);
    fsa << contents;
    fsa.close();
    //#endif
}

string FileSystem::get_file_name(string str, bool extension)
{
    vector<string> delimeter, remove;
    delimeter.push_back("/");
    delimeter.push_back("\\");
    delimeter.push_back("//");
    vector<string> splits1 = FileSystem::split(str, delimeter, remove);
    if (extension == true)
    {
        return splits1[splits1.size() - 1];
    }
    else
    {
        vector<string> splits = FileSystem::split(splits1[splits1.size() - 1], ".");
        string ext = splits[splits.size() - 1];
        string result = splits1[splits1.size() - 1];
        result.erase(result.length() - (ext.length() + 1), ext.length() + 1);
        return result;
    }
}

string FileSystem::get_file_extension(string str) {
    return str.substr(str.find_last_of(".") + 1);
}


void FileSystem::make_folder(string path)
{
    //CreateDirectory(path.c_str(), NULL);
#if defined(_MSC_VER)
    std::filesystem::create_directory(path.c_str());
#else
    std::filesystem::create_directory(path);
    // char tmp[512] = { 0, };
    // sprintf(tmp, "mkdir -p '%s'", path.c_str());
    // system(tmp);
#endif
}

string FileSystem::get_folder_name(string str)
{
    vector<string> delimeter, remove;
    delimeter.push_back("/");
    delimeter.push_back("\\");
    delimeter.push_back("//");
    vector<string> splits = FileSystem::split(str, delimeter, remove);
    string result = "";
    for (int i = 0; i < splits.size() - 1; i++)
    {
        result.append(splits[i]);
        result.append("//");
    }
#ifdef _WIN32
    return result;
#else
    return "/" + result;
#endif
}

bool FileSystem::exist(string path)
{
#ifdef _WIN32
    return std::filesystem::exists(path);
#elif defined(__APPLE__)
    return std::filesystem::exists(path);
#else
    return std::filesystem::exists(path);
    /*if (FILE* file = fopen(path.c_str(), "r"))
    {
        fclose(file);
        return true;
    }
    else return false;*/
#endif
}

void FileSystem::remove_file(string path)
{
#ifdef _WIN32
    fs::remove(path.c_str());
#elif defined(__APPLE__)
    fs::remove(path.c_str());
#else
    fs::remove(path.c_str());
#endif
}

void FileSystem::remove_directory(string path)
{
#ifdef _WIN32
    fs::remove_all(path.c_str());
#elif defined(__APPLE__)
    fs::remove_all(path.c_str());
#else
    // fs::remove_all(path.c_str());
    std::filesystem::remove_all(path);
#endif
}

void FileSystem::rename_file(string origin_path, string target_path)
{
#ifdef _WIN32
    rename(origin_path.c_str(), target_path.c_str());
#elif defined(__APPLE__)
    rename(origin_path.c_str(), target_path.c_str());
#else
    rename(origin_path.c_str(), target_path.c_str());
#endif
}

void FileSystem::copy_file(string origin_path, string target_path)
{
    fs::copy(origin_path.c_str(), target_path.c_str());
}

bool FileSystem::ends_with(string const& value, string const& ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

string FileSystem::replace_file_name(string const& origin_path, string const& target_word, string const& replace_word) {
    string result = origin_path;
    size_t start_pos = origin_path.find(target_word);
    if (start_pos != std::string::npos) {
        result.replace(start_pos, target_word.length(), replace_word);
    }
    return result;
}
