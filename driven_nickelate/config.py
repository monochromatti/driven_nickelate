from pathlib import Path


class ProjectPaths:
    def __init__(self, root_path: str | Path):
        self.root: Path = Path(root_path)

    @property
    def raw_data(self) -> Path:
        return self.root / "raw_data"

    @property
    def processed_data(self) -> Path:
        return self.root / "processed_data"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def file_lists(self) -> Path:
        return self.root / "file_lists"


paths = ProjectPaths(Path(__file__).parent.resolve())
