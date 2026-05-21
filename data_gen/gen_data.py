from app.tools.gen_data import convert, set_field_size_limit, slugify

__all__ = ["set_field_size_limit", "slugify", "convert"]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        from app.tools import gen_data

        print(gen_data.__doc__)
        sys.exit(1)

    csv_input = sys.argv[1]
    out_folder = sys.argv[2] if len(sys.argv) > 2 else "tsp_json_output"
    convert(csv_input, out_folder)
