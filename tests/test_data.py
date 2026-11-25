from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from delivery_eta.data import DataError, file_sha256, load_orders, load_training_orders

from .conftest import raw_row

WriteCsv = Callable[..., Path]


def one(path: Path, **kwargs: bool):
    orders, report = load_orders(path, with_target=kwargs.get("with_target", True))
    return orders.iloc[0], report


def test_a_normal_row_is_cleaned(write_csv: WriteCsv) -> None:
    order, report = one(write_csv(raw_row()))
    assert order["order_id"] == "0x1"
    assert order["minutes"] == 24
    assert order["distance_km"] == pytest.approx(3.03, abs=0.02)
    assert order["order_hour"] == 19
    assert order["weather"] == "sunny"
    assert order["traffic"] == "jam"
    assert order["city"] == "metropolitan"
    assert order["vehicle"] == "electric_scooter"
    assert order["festival"] == "no"
    assert order["problem"] == ""
    assert report.rows_usable == 1


def test_flipped_signs_are_put_right(write_csv: WriteCsv) -> None:
    flipped = raw_row(Restaurant_latitude="-22.745049", Restaurant_longitude="-75.892471")
    order, report = one(write_csv(flipped))
    assert order["distance_km"] == pytest.approx(3.03, abs=0.02)
    assert report.fixed["flipped coordinate signs"] == 1


def test_unrecorded_locations_are_rejected_with_a_reason(write_csv: WriteCsv) -> None:
    row = raw_row(
        Restaurant_latitude="0.0",
        Restaurant_longitude="0.0",
        Delivery_location_latitude="0.11",
        Delivery_location_longitude="0.11",
    )
    order, report = one(write_csv(row))
    assert order["problem"] == "no location recorded"
    assert report.rejected == {"no location recorded": 1}


def test_missing_order_time_is_estimated_from_pickup(write_csv: WriteCsv) -> None:
    order, _ = one(write_csv(raw_row(Time_Orderd="NaN ", Time_Order_picked="00:05:00")))
    assert order["order_hour"] == 23


@pytest.mark.parametrize(
    "change", [{"Delivery_person_Age": "15"}, {"Delivery_person_Ratings": "6"}]
)
def test_impossible_rider_details_are_not_trusted(
    write_csv: WriteCsv, change: dict[str, str]
) -> None:
    order, report = one(write_csv(raw_row(**change)))
    assert np.isnan(order["rider_age"])
    assert np.isnan(order["rider_rating"])
    assert report.fixed["impossible rider details blanked"] == 1


def test_blanks_and_unknown_words_become_missing(write_csv: WriteCsv) -> None:
    row = raw_row(
        Weatherconditions="conditions NaN",
        Festival="NaN ",
        Road_traffic_density="Gridlock",
        Vehicle_condition="3",
    )
    order, report = one(write_csv(row))
    assert order[["weather", "festival", "traffic", "vehicle_condition"]].isna().all()
    assert report.fixed["unknown traffic blanked"] == 1


def test_bad_targets_and_repeats_are_rejected(write_csv: WriteCsv) -> None:
    path = write_csv(
        raw_row(),
        raw_row(),
        raw_row(ID="0x2", **{"Time_taken(min)": "(min) 0"}),
        raw_row(ID="0x3", **{"Time_taken(min)": "soon"}),
    )
    orders, report = load_orders(path, with_target=True)
    assert report.rejected == {"no usable delivery time": 2, "repeated order": 1}
    assert orders["problem"].tolist()[0] == ""


def test_the_target_is_optional_when_predicting(write_csv: WriteCsv) -> None:
    row = raw_row()
    del row["Time_taken(min)"]
    order, _ = one(write_csv(row), with_target=False)
    assert order["problem"] == ""
    with pytest.raises(DataError, match=r"Time_taken"):
        load_orders(write_csv(row), with_target=True)


def test_problems_with_the_file_are_explained(tmp_path: Path, write_csv: WriteCsv) -> None:
    with pytest.raises(DataError, match="No file"):
        load_orders(tmp_path / "nothing.csv", with_target=True)
    row = raw_row()
    del row["City"]
    with pytest.raises(DataError, match="City"):
        load_orders(write_csv(row), with_target=True)
    with pytest.raises(DataError, match="at least 1000"):
        load_training_orders(write_csv(raw_row()))


def test_training_refuses_a_file_that_is_mostly_unusable(write_csv: WriteCsv) -> None:
    good = [raw_row(ID=f"a{i}") for i in range(10)]
    bad = [raw_row(ID=f"b{i}", Restaurant_latitude="0") for i in range(10)]
    with pytest.raises(DataError, match="Too many rows rejected"):
        load_training_orders(write_csv(*good, *bad), min_rows=5)


def test_fingerprint_ignores_line_endings(tmp_path: Path) -> None:
    unix, windows = tmp_path / "u.csv", tmp_path / "w.csv"
    unix.write_bytes(b"a,b\n1,2\n")
    windows.write_bytes(b"a,b\r\n1,2\r\n")
    assert file_sha256(unix) == file_sha256(windows)
