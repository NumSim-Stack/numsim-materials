// Replaces the global operator new with one that counts its calls (test_system_allocations).
#include <cstddef>
#include <cstdlib>
#include <new>

namespace numsim_test {
namespace {
std::size_t calls{0};
}
std::size_t new_calls() { return calls; }
} // namespace numsim_test

void *operator new(std::size_t size) {
  ++numsim_test::calls;
  if (void *p{std::malloc(size == 0 ? 1 : size)})
    return p;
  throw std::bad_alloc{};
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
// std::get_temporary_buffer (stable_sort) allocates with the nothrow forms: they must pair with the delete above.
void *operator new(std::size_t size, std::nothrow_t const &) noexcept {
  ++numsim_test::calls;
  return std::malloc(size == 0 ? 1 : size);
}
void *operator new[](std::size_t size, std::nothrow_t const &tag) noexcept { return ::operator new(size, tag); }
void operator delete(void *p, std::nothrow_t const &) noexcept { std::free(p); }
void operator delete[](void *p, std::nothrow_t const &) noexcept { std::free(p); }
