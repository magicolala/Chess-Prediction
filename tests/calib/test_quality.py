import textwrap

from oracle.calib.quality import classify_quality, load_thresholds


def test_classify_quality_uses_default_thresholds():
    thresholds = load_thresholds(env={})

    assert classify_quality(50, 45, thresholds) == "good"
    assert classify_quality(50, 0, thresholds) == "inaccuracy"
    assert classify_quality(50, -120, thresholds) == "mistake"
    assert classify_quality(50, -400, thresholds) == "blunder"


def test_classify_quality_reads_env():
    env = {
        "ORACLE_QUALITY_GOOD": "0",
        "ORACLE_QUALITY_INACCURACY": "10",
        "ORACLE_QUALITY_MISTAKE": "30",
        "ORACLE_QUALITY_BLUNDER": "60",
    }

    thresholds = load_thresholds(env=env)

    assert classify_quality(0, -5, thresholds) == "inaccuracy"
    assert classify_quality(0, -20, thresholds) == "mistake"
    assert classify_quality(0, -100, thresholds) == "blunder"


def test_classify_quality_reads_yaml(tmp_path):
    yaml_content = textwrap.dedent(
        """
        quality:
          good: 0
          inaccuracy: 15
          mistake: 40
          blunder: 100
        """
    )
    yaml_file = tmp_path / "quality.yaml"
    yaml_file.write_text(yaml_content)

    thresholds = load_thresholds(env={}, yaml_path=str(yaml_file))

    assert classify_quality(0, -10, thresholds) == "inaccuracy"
    assert classify_quality(0, -30, thresholds) == "mistake"
    assert classify_quality(0, -90, thresholds) == "blunder"
