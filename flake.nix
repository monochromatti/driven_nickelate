{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
    fenix = {
      url = "github:nix-community/fenix";
      inputs = { nixpkgs.follows = "nixpkgs"; };
    };
  };

  outputs = inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
      ];
      systems = nixpkgs.lib.systems.flakeExposed;
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        devenv.shells.default = {
          packages = with pkgs; [ maturin quarto ];
          languages.python = {
            enable = true;
            uv = {
              enable = true;
              sync.enable = true;
            };
          };
          languages.rust = {
            enable = true;
            channel = "nightly";
          };
          # enterShell = ''
          #   uvx maturin build --manifest-path ./packages/polars-splines/Cargo.toml
          # '';
        };
      };
    };
}
