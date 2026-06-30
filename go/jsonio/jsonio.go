package jsonio

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/redevops-io/context-runtime/crtypes"
)

var rawMessageType = reflect.TypeOf(json.RawMessage{})

// Marshal serializes Context Runtime boundary values to their stable JSON form.
//
// Struct fields are emitted with their json tag names. If a struct has an
// Extra field tagged as "extra", its entries are merged into the surrounding
// object instead of being nested under an "extra" key. Values stored as
// json.RawMessage are written back verbatim by encoding/json, preserving
// forward-compatible unknown fields across a load/save round-trip.
func Marshal(v any) ([]byte, error) {
	jsonValue, err := dumpValue(reflect.ValueOf(v))
	if err != nil {
		return nil, err
	}
	return json.Marshal(jsonValue)
}

// Unmarshal decodes stable JSON into v.
//
// Unknown object fields on structs that expose an Extra map are preserved in
// that map as json.RawMessage values. Payloads carrying a spec_version whose
// major version is greater than crtypes.SpecVersion are rejected before decode.
func Unmarshal(data []byte, v any) error {
	rv := reflect.ValueOf(v)
	if !rv.IsValid() || rv.Kind() != reflect.Pointer || rv.IsNil() {
		// Let encoding/json return its standard InvalidUnmarshalError.
		return json.Unmarshal(data, v)
	}

	if err := checkSpecVersionForType(rv.Type(), json.RawMessage(data)); err != nil {
		return err
	}
	if err := json.Unmarshal(data, v); err != nil {
		return err
	}
	return populateExtras(rv, json.RawMessage(data))
}

func dumpValue(v reflect.Value) (any, error) {
	if !v.IsValid() {
		return nil, nil
	}

	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return nil, nil
		}
		return dumpValue(v.Elem())
	}

	if v.Type() == rawMessageType {
		return v.Interface(), nil
	}

	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return nil, nil
		}
		return dumpValue(v.Elem())
	}

	switch v.Kind() {
	case reflect.Struct:
		return dumpStruct(v)
	case reflect.Slice:
		if v.IsNil() {
			return nil, nil
		}
		if v.Type().Elem().Kind() == reflect.Uint8 {
			// Preserve encoding/json's base64 behavior for []byte values.
			return v.Interface(), nil
		}
		fallthrough
	case reflect.Array:
		out := make([]any, v.Len())
		for i := 0; i < v.Len(); i++ {
			item, err := dumpValue(v.Index(i))
			if err != nil {
				return nil, err
			}
			out[i] = item
		}
		return out, nil
	case reflect.Map:
		if v.IsNil() {
			return nil, nil
		}
		if v.Type().Key().Kind() != reflect.String {
			return v.Interface(), nil
		}
		out := make(map[string]any, v.Len())
		for _, key := range v.MapKeys() {
			item, err := dumpValue(v.MapIndex(key))
			if err != nil {
				return nil, err
			}
			out[key.String()] = item
		}
		return out, nil
	default:
		return v.Interface(), nil
	}
}

func dumpStruct(v reflect.Value) (map[string]any, error) {
	fields := jsonFields(v.Type())
	out := make(map[string]any, len(fields))

	for name, field := range fields {
		if name == "extra" {
			continue
		}
		fv := v.FieldByIndex(field.index)
		item, err := dumpValue(fv)
		if err != nil {
			return nil, err
		}
		out[name] = item
	}

	if field, ok := fields["extra"]; ok {
		fv := v.FieldByIndex(field.index)
		if fv.Kind() == reflect.Map && !fv.IsNil() && fv.Type().Key().Kind() == reflect.String {
			for _, key := range fv.MapKeys() {
				name := key.String()
				if _, exists := out[name]; exists {
					continue
				}
				item, err := dumpValue(fv.MapIndex(key))
				if err != nil {
					return nil, err
				}
				out[name] = item
			}
		}
	}

	return out, nil
}

func populateExtras(v reflect.Value, raw json.RawMessage) error {
	if !v.IsValid() {
		return nil
	}
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return nil
		}
		return populateExtras(v.Elem(), raw)
	}
	if v.Type() == rawMessageType {
		return nil
	}
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return nil
		}
		return populateExtras(v.Elem(), raw)
	}

	switch v.Kind() {
	case reflect.Struct:
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(raw, &obj); err != nil || obj == nil {
			return nil
		}

		fields := jsonFields(v.Type())
		extra := make(map[string]json.RawMessage)
		if rawExtra, ok := obj["extra"]; ok {
			var nested map[string]json.RawMessage
			if err := json.Unmarshal(rawExtra, &nested); err == nil {
				for key, val := range nested {
					extra[key] = val
				}
			}
		}

		for name, val := range obj {
			if name == "extra" {
				continue
			}
			field, known := fields[name]
			if !known {
				extra[name] = val
				continue
			}
			if err := populateExtras(v.FieldByIndex(field.index), val); err != nil {
				return err
			}
		}

		if field, ok := fields["extra"]; ok {
			return setExtraMap(v.FieldByIndex(field.index), extra)
		}
		return nil
	case reflect.Slice:
		if v.IsNil() || v.Type().Elem().Kind() == reflect.Uint8 {
			return nil
		}
		fallthrough
	case reflect.Array:
		var items []json.RawMessage
		if err := json.Unmarshal(raw, &items); err != nil {
			return nil
		}
		limit := v.Len()
		if len(items) < limit {
			limit = len(items)
		}
		for i := 0; i < limit; i++ {
			if err := populateExtras(v.Index(i), items[i]); err != nil {
				return err
			}
		}
		return nil
	case reflect.Map:
		if v.IsNil() || v.Type().Key().Kind() != reflect.String {
			return nil
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(raw, &obj); err != nil || obj == nil {
			return nil
		}
		for _, key := range v.MapKeys() {
			rawVal, ok := obj[key.String()]
			if !ok {
				continue
			}
			val := v.MapIndex(key)
			if !val.IsValid() {
				continue
			}
			if val.Kind() == reflect.Pointer || val.Kind() == reflect.Interface {
				if err := populateExtras(val, rawVal); err != nil {
					return err
				}
				continue
			}
			if val.Kind() == reflect.Struct || val.Kind() == reflect.Slice || val.Kind() == reflect.Array || val.Kind() == reflect.Map {
				copyVal := reflect.New(val.Type()).Elem()
				copyVal.Set(val)
				if err := populateExtras(copyVal, rawVal); err != nil {
					return err
				}
				v.SetMapIndex(key, copyVal)
			}
		}
	}
	return nil
}

func setExtraMap(field reflect.Value, extra map[string]json.RawMessage) error {
	if !field.IsValid() || !field.CanSet() || field.Kind() != reflect.Map || field.Type().Key().Kind() != reflect.String {
		return nil
	}
	if len(extra) == 0 {
		field.Set(reflect.Zero(field.Type()))
		return nil
	}

	out := reflect.MakeMapWithSize(field.Type(), len(extra))
	for key, raw := range extra {
		val, err := rawMessageMapValue(field.Type().Elem(), raw)
		if err != nil {
			return err
		}
		out.SetMapIndex(reflect.ValueOf(key).Convert(field.Type().Key()), val)
	}
	field.Set(out)
	return nil
}

func rawMessageMapValue(elem reflect.Type, raw json.RawMessage) (reflect.Value, error) {
	rawVal := reflect.ValueOf(raw)
	if rawVal.Type().AssignableTo(elem) {
		return rawVal, nil
	}
	if rawVal.Type().ConvertibleTo(elem) {
		return rawVal.Convert(elem), nil
	}
	if elem.Kind() == reflect.Interface && rawVal.Type().Implements(elem) {
		return rawVal, nil
	}

	decoded := reflect.New(elem)
	if err := json.Unmarshal(raw, decoded.Interface()); err != nil {
		return reflect.Value{}, err
	}
	return decoded.Elem(), nil
}

func checkSpecVersionForType(t reflect.Type, raw json.RawMessage) error {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == rawMessageType {
		return nil
	}

	switch t.Kind() {
	case reflect.Struct:
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(raw, &obj); err != nil || obj == nil {
			return nil
		}
		if sv, ok := obj["spec_version"]; ok {
			if err := rejectHigherMajor(sv); err != nil {
				return err
			}
		}
		fields := jsonFields(t)
		for name, field := range fields {
			if name == "extra" {
				continue
			}
			if val, ok := obj[name]; ok {
				if err := checkSpecVersionForType(field.typ, val); err != nil {
					return err
				}
			}
		}
	case reflect.Slice, reflect.Array:
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			return nil
		}
		var items []json.RawMessage
		if err := json.Unmarshal(raw, &items); err != nil {
			return nil
		}
		for _, item := range items {
			if err := checkSpecVersionForType(t.Elem(), item); err != nil {
				return err
			}
		}
	case reflect.Map:
		if t.Key().Kind() != reflect.String || t.Elem().Kind() == reflect.Interface {
			return nil
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(raw, &obj); err != nil {
			return nil
		}
		for _, val := range obj {
			if err := checkSpecVersionForType(t.Elem(), val); err != nil {
				return err
			}
		}
	}
	return nil
}

func rejectHigherMajor(raw json.RawMessage) error {
	var specVersion string
	if err := json.Unmarshal(raw, &specVersion); err != nil {
		return nil
	}
	major, err := majorVersion(specVersion)
	if err != nil {
		return fmt.Errorf("invalid spec_version %q: %w", specVersion, err)
	}
	supported, err := majorVersion(crtypes.SpecVersion)
	if err != nil {
		return fmt.Errorf("invalid supported spec_version %q: %w", crtypes.SpecVersion, err)
	}
	if major > supported {
		return fmt.Errorf("spec_version %s has a higher major than supported %s", specVersion, crtypes.SpecVersion)
	}
	return nil
}

func majorVersion(version string) (int, error) {
	major, _, _ := strings.Cut(version, ".")
	return strconv.Atoi(major)
}

type fieldInfo struct {
	index []int
	typ   reflect.Type
}

func jsonFields(t reflect.Type) map[string]fieldInfo {
	fields := make(map[string]fieldInfo, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if field.PkgPath != "" {
			continue
		}
		name, ok := jsonFieldName(field)
		if !ok {
			continue
		}
		if _, exists := fields[name]; exists {
			continue
		}
		fields[name] = fieldInfo{index: field.Index, typ: field.Type}
	}
	return fields
}

func jsonFieldName(field reflect.StructField) (string, bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false
	}
	name, _, _ := strings.Cut(tag, ",")
	if name == "" {
		name = field.Name
	}
	return name, true
}
