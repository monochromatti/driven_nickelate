{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
  };

  outputs =
    inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
      ];
      systems = nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, ... }:
        let
          python = pkgs.python311;
          buildInputs = with pkgs; [
            stdenv.cc.cc
            libuv
            zlib
            cairo
          ];
          LIBRARY_PATH = if pkgs.stdenv.isDarwin then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH";
        in
        {
          devenv.shells.default = {
            packages = with pkgs; [
              maturin
              quarto
              python.pkgs.jupyter
            ];
            languages.python = {
              enable = true;
              package = python;
              uv = {
                enable = true;
                sync.enable = true;
              };
            };
            env = {
              LIBRARY_PATH = "${with pkgs; lib.makeLibraryPath buildInputs}";
              UV_PYTHON = "${python}/bin/python";
            };
            enterShell = ''
              echo "🔬 Driven nickelate project shell"
            '';
          };
        };
    };
}
