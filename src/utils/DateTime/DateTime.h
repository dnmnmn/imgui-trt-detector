#ifndef __DATE_TIME_H__
#define __DATE_TIME_H__

#include <string>
#include <memory>
using namespace std;

class DateTime
{
private:
    tm t;
public:
    DateTime(string str = "1970-01-01 00:00:00");
    ~DateTime();
public:
    bool operator >(const DateTime& dt);
    bool operator <(const DateTime& dt);
    bool operator >=(const DateTime& dt);
    bool operator <=(const DateTime& dt);
    bool operator ==(const DateTime& dt);
public:
    void set_time_from_string(string str = "1970-01-01 00:00:00");
    void now();
    void add_year(int value);
    void add_month(int value);
    void add_day(int value);
    void add_hour(int value);
    void add_min(int value);
    void add_sec(int value);
    int get_hour();
    string get_string();
    string get_continual_string();
    string get_date_string();
    string get_log_string();
};

#endif