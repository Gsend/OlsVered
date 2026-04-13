fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();
    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file("include/olssm.h");
        }
        Err(e) => {
            // C header generation is optional — only needed for C/C++ consumers.
            // The Python wheel and Rust library build fine without it.
            eprintln!("cargo:warning=cbindgen could not generate C bindings: {e}");
        }
    }
}
