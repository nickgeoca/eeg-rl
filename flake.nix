# NOTE: This is a personal dev-shell flake — it is NOT part of the source tree.
{
  description = "eeg-rl dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python3.withPackages (ps: [
        ps.numpy
        ps.gymnasium
        ps.plotext
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [ python ];
      };
    };
}
