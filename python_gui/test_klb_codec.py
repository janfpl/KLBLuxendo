from __future__ import annotations

import filecmp
import json
import tempfile
import zipfile
from pathlib import Path

import h5py
import klb_codec as klb
import numpy as np


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    sample_raw = repo / "testData" / "img.raw"
    sample_klb = repo / "testData" / "img.klb"

    header = klb.read_klb_header(sample_klb)
    assert header.xyzct == (101, 151, 29, 1, 1)
    assert header.dtype_name == "uint16"
    assert header.compression_name == "bzip2"

    with tempfile.TemporaryDirectory(prefix=".klb_gui_test_", dir=repo) as temp_dir:
        temp = Path(temp_dir)
        decompressed = temp / "decompressed.raw"
        klb.decompress_klb_to_raw(sample_klb, decompressed, workers=2)
        assert filecmp.cmp(sample_raw, decompressed, shallow=False)

        recompressed = temp / "recompressed.klb"
        klb.compress_raw_to_klb(
            sample_raw,
            recompressed,
            xyzct=header.xyzct,
            data_type=header.dtype_name,
            pixel_size=header.pixel_size,
            block_size=header.block_size,
            compression=header.compression_name,
            metadata="roundtrip test",
            workers=2,
        )
        roundtrip = temp / "roundtrip.raw"
        klb.decompress_klb_to_raw(recompressed, roundtrip, workers=2)
        assert filecmp.cmp(sample_raw, roundtrip, shallow=False)

        lux_source = temp / "tiny.lux.h5"
        data = (np.arange(5 * 7 * 11, dtype=np.uint16).reshape(5, 7, 11) * 3) % 65535
        metadata = {
            "processingInformation": {
                "voxel_size_um": {"width": 1.5, "height": 2.5, "depth": 3.5},
                "image_size_vx": {"width": 11, "height": 7, "depth": 5},
            }
        }
        with h5py.File(lux_source, "w") as h5_file:
            h5_file.create_dataset("Data", data=data, chunks=(2, 3, 4))
            h5_file.create_dataset("Data_2_2_2", data=data[::2, ::2, ::2], chunks=(1, 2, 3))
            h5_file.create_dataset("metadata", data=json.dumps(metadata), dtype=h5py.string_dtype("utf-8"))

        lux_bundle = temp / "tiny_klb_bundle"
        lux_result = klb.compress_hdf5_all_layers_to_klb_bundle(lux_source, lux_bundle, block_size=(4, 3, 2, 1, 1), workers=2)
        assert lux_result.layer_count == 2
        assert (lux_bundle / klb.KLB_BUNDLE_MANIFEST).exists()

        lux_restore_dir = temp / "tiny_restore"
        restored = klb.decompress_klb_bundle_to_lux_folder(lux_bundle, lux_restore_dir, workers=2)
        assert restored == [lux_restore_dir / "tiny.lux.h5"]
        with h5py.File(restored[0], "r") as h5_file:
            np.testing.assert_array_equal(h5_file["Data"][...], data)
            np.testing.assert_array_equal(h5_file["Data_2_2_2"][...], data[::2, ::2, ::2])
            restored_metadata = h5_file["metadata"][()]
            if isinstance(restored_metadata, bytes):
                restored_metadata = restored_metadata.decode("utf-8")
            assert json.loads(restored_metadata) == metadata

        batch_source = temp / "experiment"
        batch_source.mkdir()
        (batch_source / "reg_obj").mkdir()
        (batch_source / "reg_sti").mkdir()
        (batch_source / "reg_obj" / "translation.txt").write_text("dx 1\n", encoding="utf-8")
        batch_lux = batch_source / "batch.lux.h5"
        with h5py.File(batch_lux, "w") as h5_file:
            h5_file.create_dataset("Data", data=data, chunks=(2, 3, 4))
            h5_file.create_dataset("Data_2_2_2", data=data[::2, ::2, ::2], chunks=(1, 2, 3))
            h5_file.create_dataset("metadata", data=json.dumps(metadata), dtype=h5py.string_dtype("utf-8"))
        (batch_source / "config_task.json").write_text('{"ok": true}', encoding="utf-8")
        batch_output = temp / "experiment_klb_bundle"
        batch_result = klb.compress_directory_to_klb_bundle(
            batch_source,
            batch_output,
            block_size=(4, 3, 2, 1, 1),
            workers=2,
        )
        assert batch_result.klb_files == 2
        assert batch_result.archived_files == 2
        assert batch_result.archived_dirs == 2
        assert (batch_output / klb.KLB_BUNDLE_MANIFEST).exists()
        assert (batch_output / "klb_layers" / "batch" / "Data.klb").exists()
        assert (batch_output / "klb_layers" / "batch" / "Data_2_2_2.klb").exists()
        assert batch_result.archive_path is not None
        with zipfile.ZipFile(batch_result.archive_path, "r") as archive:
            assert sorted(archive.namelist()) == ["config_task.json", "reg_obj/", "reg_obj/translation.txt", "reg_sti/"]
        batch_restore = temp / "experiment_restore"
        restored_files = klb.decompress_klb_bundle_to_lux_folder(batch_output, batch_restore, workers=2)
        assert restored_files == [batch_restore / "batch.lux.h5"]
        assert (batch_restore / "config_task.json").read_text(encoding="utf-8") == '{"ok": true}'
        assert (batch_restore / "reg_obj").is_dir()
        assert (batch_restore / "reg_obj" / "translation.txt").read_text(encoding="utf-8") == "dx 1\n"
        assert (batch_restore / "reg_sti").is_dir()
        with h5py.File(restored_files[0], "r") as h5_file:
            np.testing.assert_array_equal(h5_file["Data"][...], data)
            np.testing.assert_array_equal(h5_file["Data_2_2_2"][...], data[::2, ::2, ::2])
            restored_metadata = h5_file["metadata"][()]
            if isinstance(restored_metadata, bytes):
                restored_metadata = restored_metadata.decode("utf-8")
            assert json.loads(restored_metadata) == metadata

    print("KLB codec sample roundtrip passed")


if __name__ == "__main__":
    main()
