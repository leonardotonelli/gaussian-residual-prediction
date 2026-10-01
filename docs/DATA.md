# Data and attribution

## Moving-MNIST

Moving-MNIST clips are generated online from MNIST digits; no video dataset is downloaded. Run:

```bash
python scripts/prepare_moving_mnist.py --data-dir data/concept2-mnist --download
```

The preparer downloads the four original MNIST archives from the public PyTorch OSSCI mirror, validates their official checksums, verifies raw IDX files, and creates the fixed seed-0 identity manifest. The split is 50,000 training, 10,000 development and 10,000 official test identities. Only the declared bank stage loads the test images. The published identity-manifest metadata is supplied separately in `assets/mnist_identity_manifest.json`; pixel data is not redistributed.

MNIST attribution: Yann LeCun, Léon Bottou, Yoshua Bengio and Patrick Haffner, *Gradient-Based Learning Applied to Document Recognition*, Proceedings of the IEEE, 1998. The official distribution page does not state a dataset license; the repository's MIT license applies to our code, not the MNIST data.

## MPI3D-realistic

Use the **realistic rendered** variant, not MPI3D-real or MPI3D-toy. Download its NPZ from the [official dataset repository](https://github.com/rr-learning/disentanglement_dataset), which links the [MPI3D-realistic archive](https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main/mpi3d_realistic.npz).

```bash
python scripts/prepare_mpi3d.py --archive /path/to/mpi3d_realistic.npz \
  --output data/mpi3d/images.npy
```

Alternatively, `python scripts/prepare_mpi3d.py --download` downloads that archive explicitly. The script streams its `images.npy` member to disk, verifies the exact preprint SHA256, and checks its shape/dtype using memory mapping. It avoids loading the full array into RAM or rewriting the NumPy header. Allow space for both the downloaded archive and the approximately 12.7 GB extracted array. Existing arrays are verified and never silently replaced.

Expected extracted file SHA256:

```text
f6124f358e02846695cb3795acf8714f990c84cf02d495796898c133cf1ae866
```

The array shape is `(1036800, 64, 64, 3)`, dtype `uint8`. Task construction fixes background index 0, changes position by four grid indices on successful commands, and balances success/failure at 50%. Attribute combinations are assigned by `(colour + shape + 3 * size) mod 6`: residues 0–3 train, 4 validation, 5 test. Position manifests are supplied under `config/shared/mpi3d_position_manifests/` and are verified by the implementation.

MPI3D is distributed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Gondal et al., *On the Transfer of Inductive Bias from Simulation to the Real World: a New Disentanglement Dataset*, NeurIPS 2019. The dataset is not redistributed here. The supplied MPI3D illustration is derived from that dataset and retains its attribution and CC BY 4.0 terms; it is excluded from the code's MIT license.

## Method and formatting attribution

The paper explains its adaptations of BYOL (Grill et al., 2020), AdaSSL (Zhang et al., 2026), LeJEPA (Balestriero and LeCun, 2025) and LeWorldModel (Maes et al., 2026). Full citations are in `paper/references.bib`. These are local implementations/adaptations, not claimed reproductions of official results.

The NeurIPS 2026 style is included unchanged from the official template for compiling the paper; its original notice is retained. The manuscript and figures are supplied as research artifacts, separate from third-party dataset terms. The MIT license covers the authors' software.
