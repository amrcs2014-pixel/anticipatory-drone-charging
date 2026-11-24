"""Download real locations for the Amman case from OpenStreetMap (Overpass API).

    python case/fetch_osm.py

Writes case/osm_amman.json: pharmacies, hospitals and fuel stations inside the
bounding box.  Data (c) OpenStreetMap contributors, ODbL.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BBOX = (31.86, 35.80, 32.05, 36.02)          # south, west, north, east
QUERY = f"""
[out:json][timeout:120];
(
  node["amenity"="pharmacy"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["amenity"="hospital"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["amenity"="hospital"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["amenity"="fuel"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["amenity"="fuel"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out center tags;
"""
SERVERS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter"]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    data = None
    for attempt in range(6):
        url = SERVERS[attempt % len(SERVERS)]
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": QUERY}).encode(),
                                         headers={"User-Agent": "research-case-builder/1.0"})
            data = json.load(urllib.request.urlopen(req, timeout=180))
            break
        except Exception as e:                      # noqa: BLE001
            print("failed", url, e, flush=True)
            time.sleep(5)
    if data is None:
        sys.exit("could not reach any Overpass server")
    out = []
    for el in data["elements"]:
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None:
            continue
        t = el.get("tags", {})
        out.append({"osm": f"{el['type']}/{el['id']}", "kind": t.get("amenity"), "lat": lat, "lon": lon,
                    "name": t.get("name:en") or t.get("name") or ""})
    json.dump({"bbox": BBOX, "timestamp": data.get("osm3s", {}).get("timestamp_osm_base"), "elements": out},
              open(os.path.join(HERE, "osm_amman.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    for k in ("pharmacy", "hospital", "fuel"):
        print(k, sum(1 for e in out if e["kind"] == k))
    print("timestamp", data.get("osm3s", {}).get("timestamp_osm_base"))


if __name__ == "__main__":
    main()
