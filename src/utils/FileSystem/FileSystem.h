#ifndef __FILE_SYSTEM_H__
#define __FILE_SYSTEM_H__

#include <string>
#include <vector>

using namespace std;

class FileSystem
{
private:
    static std::string last_dialog_path;
public:
    static std::string get_app_path();
    // https://github.com/mlabbe/nativefiledialog
    static bool open_file_dialog(std::string filter, std::string default_path, std::string* result);
    static bool open_multiple_file_dialog(std::string filter, std::string default_path, vector<std::string>* result);
    static bool save_file_dialog(std::string filter, std::string default_path, std::string* result);
    static bool open_folder_dialog(std::string default_path, std::string* result);
    static bool save_folder_dialog(std::string default_path, std::string* result);
    static string get_dialog_last_path();
public:
    static std::vector<string> get_directories(string dir_path);
    static std::vector<string> get_files(string dir_path, string filter);
    static string get_file_name(string str, bool extension = false);
    static string get_file_extension(string str);       // [exe, jpg, png, ...]
    static string get_folder_name(string str);
    static void make_folder(string path);
    static bool exist(string path);
    static void remove_file(string path);
    static void remove_directory(string path);
    static void rename_file(string origin_path, string target_path);
    static void copy_file(string origin_path, string target_path);
    static string replace_file_name(string const& origin_path, string const& target_word, string const& replace_word);
public:
    static std::vector<string> read_all_textline(string path);
    static void write_all_text(string path, string contents);
    static void write_add_text(string path, string contents);
public:
    static std::vector<string> split(string stringToBeSplitted, string delimeter);
    static vector<string> split(string stringToBeSplitted, vector<string> delimeters, vector<string> remove);
public:
    static bool ends_with(string const & value, string const & ending);
};

#endif