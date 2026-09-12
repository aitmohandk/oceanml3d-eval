# External products

One manifest per external product (same format as `oceanml3d-core` exports, see
`docs/product_format.md`). `${OCEANML3D_DATA}` is expanded at load time. To split a yearly
file into daily files: `oceanml3d-eval split --input big.nc --out dir --name duacs` (Phase 2)
or simply set `pattern` to match the single file (no date groups → no date filtering).
