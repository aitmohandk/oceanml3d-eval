# External products

One manifest per external product (same format as `oceanml3d-core` exports, see
`docs/product_format.md`). `${OCEANML3D_DATA}` is expanded at load time. To split a yearly
file into daily files: `oceanml3d-eval split --input big.nc --out dir --name duacs` (Phase 2)
or simply set `pattern` to match the single file (no date groups → no date filtering).

## 3D references

A truth with a depth axis becomes a product with **one variable per level**, named `<var>_d<ii>`
after its *position on the file's depth axis* -- the same `depth_index` that `oceanml3d-core`'s
task config uses, and what the `_d<ii>` suffix means in the product format:

```bash
oceanml3d-eval split --input <store>.zarr --out <dir> --name <name> \
    --var ssh=zos --var thetao=thetao --var u=uo --var v=vo \
    --depth-indices 0,2,4,6,8,10-25          # also: '0,2,4' or 'all'
```

Variables without a depth axis (here `zos`) keep their plain canonical name. Each level carries
`depth_index` and `depth_m` in the manifest and in the NetCDF attributes.
`glorys_gs21_truth.yaml` is that manifest for the OSSE-3D benchmark, with `${OCEANML3D_DATA}`
left symbolic; `benchmarks/osse3d_gs21.yaml` scores against it, and a test checks that the two
lists of 64 variables agree.
