# Product format

`oceanml3d-core/docs/product_format.md` is the human-readable source of truth (version 1).
The machine-readable contract is `oceanml3d_eval/product_contract.py`, a byte-for-byte copy of
`oceanml3d-core/oceanml3d/inference/product_contract.py`:

    product_contract.py sha256 = dd44f0e198d0122b88207509801ed5c60a7e2fa36172151d89d78a622fe88846

`tests/test_product_contract.py` enforces the hash. Any format change must land in both repos
together (edit the file, bump `PRODUCT_FORMAT_VERSION`, update both hashes).
