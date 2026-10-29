// Stand-in for the one Boost component piranha uses (path handling), on std::filesystem.
#pragma once
#include <filesystem>

namespace boost::filesystem {
class path : public std::filesystem::path {
public:
  path() = default;
  path(const std::filesystem::path &p) : std::filesystem::path(p) {}
  path(const std::string &s) : std::filesystem::path(s) {}
  path(const char *s) : std::filesystem::path(s) {}
  bool is_complete() const { return is_absolute(); }
  friend path operator/(const path &a, const path &b) {
    return path(static_cast<const std::filesystem::path &>(a) /
                static_cast<const std::filesystem::path &>(b));
  }
};
inline bool exists(const path &p) { return std::filesystem::exists(p); }
inline bool equivalent(const path &a, const path &b) {
  std::error_code ec;
  return std::filesystem::equivalent(a, b, ec) ||
         a.lexically_normal() == b.lexically_normal();
}
} // namespace boost::filesystem
