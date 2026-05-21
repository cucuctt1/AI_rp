from app.paths import OUTPUT_ROOT, PROJECT_ROOT
from utils.exporter import Exporter


def test_default_exporter_writes_inside_new_refract():
    exporter = Exporter()

    assert str(exporter.output_root) == str(OUTPUT_ROOT)
    assert str(OUTPUT_ROOT).startswith(str(PROJECT_ROOT))
