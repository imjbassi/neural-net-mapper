from src.main import main, parse_args


def test_default_args():
    args = parse_args([])
    assert args.epochs == 60
    assert args.hidden_sizes == [128, 64]
    assert args.dropout == 0.3
    assert args.format == "mp4"
    assert not args.render_only


def test_custom_args():
    args = parse_args([
        "--epochs", "10",
        "--hidden-sizes", "32", "16",
        "--dropout", "0.5",
        "--format", "gif",
        "--seed", "7",
        "--quiet",
    ])
    assert args.epochs == 10
    assert args.hidden_sizes == [32, 16]
    assert args.dropout == 0.5
    assert args.format == "gif"
    assert args.seed == 7
    assert args.quiet


def test_difficulty_and_style_args():
    args = parse_args([
        "--uncentered", "--outline", "--jitter", "4", "--thickness", "1",
        "--theme", "dark", "--smooth", "4", "--dpi", "150",
    ])
    assert args.uncentered
    assert args.outline
    assert args.jitter == 4
    assert args.thickness == 1
    assert args.theme == "dark"
    assert args.smooth == 4
    assert args.dpi == 150


def test_render_only_without_snapshots_fails(tmp_path):
    exit_code = main(["--render-only", "--output-dir", str(tmp_path / "missing")])
    assert exit_code == 1


def test_end_to_end_tiny_run(tmp_path):
    out_dir = str(tmp_path / "out")
    exit_code = main([
        "--epochs", "2",
        "--sample-every", "2",
        "--hidden-sizes", "8",
        "--n-per-class", "8",
        "--format", "gif",
        "--output-dir", out_dir,
        "--quiet",
    ])
    assert exit_code == 0
    assert (tmp_path / "out" / "snapshots.npz").exists()
    assert (tmp_path / "out" / "animation.gif").exists()

    # Re-render from the saved snapshots
    exit_code = main(["--render-only", "--format", "gif", "--output-dir", out_dir])
    assert exit_code == 0
