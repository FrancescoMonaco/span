{
  description = "PUFFINN - Parameterless and Universal Fast FInding of Nearest Neighbors";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (system:
        f {
          pkgs = import nixpkgs {inherit system;};
        });
  in {
    devShells = forEachSupportedSystem ({pkgs}: {
      default =(pkgs.mkShell.override { stdenv = pkgs.clangStdenv; }) {
        venvDir = ".venv";
        
        packages = with pkgs; [
          clang-tools
          clang-analyzer
          lldb
          python312
          sqlite-interactive
          cmake
          just
          bear # To generate compile_commands.json files
          llvmPackages.openmp
          llvmPackages.libcxx
          rr
          gdbgui
          valgrind
          highfive
          samply
        ] ++ ( with python312Packages; [
          venvShellHook
          numpy
          h5py
          nanobind
          icecream
          scikit-build-core
        ]);

        NIX_ENFORCE_NO_NATIVE = false;
      };
    });
  };
}
