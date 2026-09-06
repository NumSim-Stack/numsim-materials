# JSON Parameter Converter — Open Issues

## Done

- ~~Replace per-type overloads with generic dispatch~~ — resolved by `type_id()` + `json_type_registry`
- ~~Remove hardcoded "name" special case~~ — "name" is in the schema via `material_base::parameters()`
- ~~Add context to conversion errors~~ — `json_type_registry::convert()` wraps exceptions with parameter name
- ~~Warn on unknown JSON keys~~ — `json_to_parameters` warns for keys in JSON but not in schema

## Remaining

### 5. Factory integration for end-to-end JSON-driven setup

The converter handles JSON → parameter_handler, but using it still requires compile-time dispatch on the material type (the if/else chain in the test). For runtime JSON-driven configuration, the factory needs to provide schemas alongside registered material types. This is a separate task.
