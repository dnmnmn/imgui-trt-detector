#include "DateTime.h"
#include <time.h>
#include <sstream>
#include <memory.h>
#include <iomanip>

#ifdef _WIN32
#define localtime_a(a,b) localtime_s(a,b)
#define sprintf_a(s,fmt,...) sprintf_s(s,fmt,__VA_ARGS__)
#else
#define localtime_a(a,b) localtime_r(b,a)
#define sprintf_a(s,fmt,...) sprintf(s,fmt,__VA_ARGS__)
#endif

DateTime::DateTime(string str)
{
    memset(&t, 0, sizeof(t));
    t.tm_mday = 1;
    set_time_from_string(str);
}

DateTime::~DateTime()
{

}

void DateTime::set_time_from_string(string str)
{
    char buffer[256] = { 0, };
    memcpy(buffer, str.c_str(), str.length());

    istringstream ss(str);
    ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");

    //strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &t);
}

void DateTime::now()
{
    time_t tn;
    time(&tn);
    localtime_a(&t, &tn);
}

int DateTime::get_hour() {
    return t.tm_hour;
}


string DateTime::get_string()
{
    char tmp[256];
    sprintf_a(tmp, "%04d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    return tmp;
}

string DateTime::get_continual_string()
{
    char tmp[256];
    sprintf_a(tmp, "%04d%02d%02d%02d%02d%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    return tmp;
}

string DateTime::get_date_string()
{
	char tmp[256];
    sprintf_a(tmp, "[%04d%02d%02d]LOG.txt", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday);
    return tmp;
}

string DateTime::get_log_string()
{
    char tmp[256];
    sprintf_a(tmp, "[%02d-%02d-%02d %02d:%02d:%02d] ", t.tm_year - 100, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    return tmp;
}

bool DateTime::operator >(const DateTime& dt)
{
    time_t a = mktime((tm*)&t);
    time_t b = mktime((tm*)&dt.t);
    if (a > b) return true;
    return false;
}

bool DateTime::operator <(const DateTime& dt)
{
    time_t a = mktime((tm*)&t);
    time_t b = mktime((tm*)&dt.t);
    if (a < b) return true;
    return false;
}

bool DateTime::operator >=(const DateTime& dt)
{
    time_t a = mktime((tm*)&t);
    time_t b = mktime((tm*)&dt.t);
    if (a >= b) return true;
    return false;
}

bool DateTime::operator <=(const DateTime& dt)
{
    time_t a = mktime((tm*)&t);
    time_t b = mktime((tm*)&dt.t);
    if (a <= b) return true;
    return false;
}

bool DateTime::operator ==(const DateTime& dt)
{
    time_t a = mktime((tm*)&t);
    time_t b = mktime((tm*)&dt.t);
    if (a == b) return true;
    return false;
}

void DateTime::add_year(int value)
{
    t.tm_year += value;
    time_t tmp = mktime(&t);
    localtime_a(&t, &tmp);
}

void DateTime::add_month(int value)
{
    t.tm_mon += value;
    time_t tmp = mktime(&t);
    localtime_a(&t, &tmp);
}

void DateTime::add_day(int value)
{
    t.tm_mday += value;
    time_t tmp = mktime(&t);
    localtime_a(&t, &tmp);
}

void DateTime::add_hour(int value)
{
    t.tm_hour += value;
    time_t tmp = mktime(&t);
    localtime_a(&t, &tmp);
}
void DateTime::add_min(int value)
{
    t.tm_min += value;
    time_t tmp = mktime(&t);
    localtime_a(&t, &tmp);
}

void DateTime::add_sec(int value)
{
    t.tm_sec += value;
    time_t tmp = mktime(&t);
    localtime_a(&t, &tmp);
}