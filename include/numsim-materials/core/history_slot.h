#ifndef NUMSIM_MATERIALS_CORE_HISTORY_SLOT_H
#define NUMSIM_MATERIALS_CORE_HISTORY_SLOT_H

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

#include <tmech/tmech.h>
#include "numsim-materials/core/history_codec.h"
#include "numsim-materials/core/history_property.h"

namespace numsim::materials {

/// One history in a flat T buffer, its value type and codec fixed where it is declared.
template <class T>
class history_slot {
public:
  using tensor2 = tmech::tensor<T, 3, 2>;

  template <class V, class Codec>
    requires history_codec_for<Codec, V, T>
  history_slot(history_property<V>& history, Codec) : m_self{std::make_unique<model<V, Codec>>(history)} {}

  std::size_t width() const { return m_self->width(); }
  void unpack(T const* src) const { m_self->unpack(src); } ///< both sides: the buffer holds the converged state
  void pack(T* dst) const { m_self->pack(dst); }             ///< the new side
  void rotate(tensor2 const& R) const { m_self->rotate(R); } ///< both sides; nothing without Codec::rotate
  bool rotates() const { return m_self->rotates(); }
  std::string_view tag() const { return m_self->tag(); }

private:
  struct concept_ {
    virtual ~concept_() = default;
    virtual std::size_t width() const = 0;
    virtual void unpack(T const* src) const = 0;
    virtual void pack(T* dst) const = 0;
    virtual void rotate(tensor2 const& R) const = 0;
    virtual bool rotates() const = 0;
    virtual std::string_view tag() const = 0;
  };

  template <class V, class Codec>
  struct model final : concept_ {
    static constexpr bool turns{requires(V const& v, tensor2 const& R) { Codec::rotate(v, R); }};

    explicit model(history_property<V>& h) : history{h}, who{h.traits().id.owner + "::" + h.traits().id.name} {}
    std::size_t width() const override { return Codec::width; }
    void unpack(T const* src) const override {
      Codec::read(history.old_value(), src);
      history.new_value() = history.old_value();
    }
    void pack(T* dst) const override { Codec::write(history.new_value(), dst, who); }
    void rotate([[maybe_unused]] tensor2 const& R) const override {
      if constexpr (turns) {
        history.old_value() = Codec::rotate(history.old_value(), R);
        history.new_value() = Codec::rotate(history.new_value(), R);
      }
    }
    bool rotates() const override { return turns; }
    std::string_view tag() const override {
      if constexpr (requires { Codec::tag; })
        return Codec::tag;
      else
        return {};
    }

    history_property<V>& history;
    std::string who;
  };

  std::unique_ptr<concept_> m_self;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_HISTORY_SLOT_H
