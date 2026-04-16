# NOTE: This is a personal dev-shell flake — it is NOT part of the source tree.
{
  description = "eeg-rl dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      cuda = pkgs.cudaPackages;
      libs = pkgs.lib.makeLibraryPath (with pkgs; [
        stdenv.cc.cc zlib zstd curl openssl bzip2 libxml2 xz systemd
        cudaPackages.cudatoolkit
        cudaPackages.cuda_cudart
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python3 uv
          cudaPackages.cudatoolkit
          cudaPackages.cuda_cudart
        ];

        shellHook = ''
          export UV_PYTHON_DOWNLOADS=never
          export LD_LIBRARY_PATH=${libs}:/run/opengl-driver/lib:$LD_LIBRARY_PATH
          uv sync --quiet
          source .venv/bin/activate
        '';
      };
    };
}
