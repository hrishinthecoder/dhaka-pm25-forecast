"""R17 - distance between the reference monitor (sensor 24434) and the one low-cost sensor that
overlaps it in time (Jahangirnagar University), from the OpenAQ location coordinates.

Needs the raw OpenAQ download (_locations.json), which is not redistributed in this repository:
    python reports/revision/r17_monitor_distance.py [raw_dir]        (default ../approach1/raw)
Great-circle (haversine) distance, mean Earth radius 6371.0088 km. The same formula reproduces
OpenAQ's distance from the query centre for the Jahangirnagar location (17.3 km)."""
import json, math, sys
from pathlib import Path

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "../approach1/raw")
OUT = Path(__file__).resolve().parent
locs = json.load(open(RAW / "_locations.json"))
CENTRE = (23.8103, 90.4125)


def km(a, b):
    R = 6371.0088
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    return 2 * R * math.asin(math.sqrt(math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def loc_of_sensor(sid):
    for l in locs:
        if sid in [s.get("id") for s in l.get("sensors", [])]:
            return l
    raise SystemExit(f"sensor {sid} not found")


mon = loc_of_sensor(24434)
ju = loc_of_sensor(11106529)
pm, pj = (mon["coordinates"]["latitude"], mon["coordinates"]["longitude"]), (ju["coordinates"]["latitude"], ju["coordinates"]["longitude"])
o = {"monitor": {"sensor_id": 24434, "location_id": mon["id"], "location_name": mon["name"], "lat": pm[0], "lon": pm[1]},
     "jahangirnagar": {"sensor_id": 11106529, "location_id": ju["id"], "location_name": ju["name"], "lat": pj[0], "lon": pj[1]},
     "km_monitor_to_jahangirnagar": round(km(pm, pj), 2),
     "km_centre_to_jahangirnagar": round(km(CENTRE, pj), 2),
     "km_centre_to_monitor": round(km(CENTRE, pm), 2)}
json.dump(o, open(OUT / "r17_monitor_distance.json", "w"), indent=2)
print(json.dumps(o, indent=1))
