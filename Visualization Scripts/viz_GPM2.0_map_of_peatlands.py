import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import rasterio
from rasterio.windows import Window
from matplotlib.colors import ListedColormap

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
R = 6371.0
lat_step = lon_step = 0.1

center    = {'lon': 0.0, 'lat': 0.0}
mouse_pos = {'lon': None, 'lat': None}

show_equator  = False
show_imagery  = False
show_peatland = False       # toggle with “p”

PEATLAND_RASTER = "data/peatGPA22WGS_2cl.tif"   # ← adjust to your path
PATCH_RADIUS_DEG = 10.0       # half-width of patch to draw (°)

# Lazy handles
peat_ds  = None              # rasterio dataset
peat_art = None              # current imshow artist for patch

# -----------------------------------------------------------------------------
def _open_peat():
    """Open the GeoTIFF once."""
    global peat_ds
    if peat_ds is None:
        peat_ds = rasterio.open(PEATLAND_RASTER)

def _get_window(lon, lat, radius_deg):
    """Return (Window, extent) around lon/lat ± radius_deg."""
    row, col = peat_ds.index(lon, lat)
    dpx_lon  = radius_deg / peat_ds.transform.a
    dpx_lat  = radius_deg / -peat_ds.transform.e
    half_x   = int(np.ceil(dpx_lon))
    half_y   = int(np.ceil(dpx_lat))

    w = Window(col - half_x, row - half_y,
               2*half_x + 1, 2*half_y + 1)
    west, north = peat_ds.transform * (w.col_off, w.row_off)
    east, south = peat_ds.transform * (w.col_off + w.width,
                                       w.row_off + w.height)
    extent = [west, east, south, north]
    return w, extent

# -----------------------------------------------------------------------------
def _draw_peat_patch(lon, lat):
    """Read & display a small peatland patch around the cursor."""
    global peat_art
    _open_peat()

    win, extent = _get_window(lon, lat, PATCH_RADIUS_DEG)
    patch = peat_ds.read(1, window=win, boundless=True, masked=True)

    # if no peat here, remove any old patch
    if np.all(patch.mask):
        if peat_art:
            peat_art.remove()
            peat_art = None
        return

    cmap = ListedColormap([(1.0, 0.55, 0.0, 0.4)])  # orange @ 40%

    if peat_art is None:
        peat_art = ax.imshow(
            patch, origin='upper', extent=extent,
            transform=ccrs.PlateCarree(), cmap=cmap,
            interpolation='nearest', zorder=3
        )
    else:
        peat_art.set_data(patch)
        peat_art.set_extent(extent)

# -----------------------------------------------------------------------------
def draw_globe():
    """Redraw the globe and (optionally) the peat patch."""
    global peat_art
    ax.clear()
    peat_art = None  # ← reset orphaned artist

    ax.projection = ccrs.Orthographic(center['lon'], center['lat'])
    ax.set_global()

    if show_imagery:
        ax.stock_img()

    if show_peatland and mouse_pos['lon'] is not None:
        _draw_peat_patch(mouse_pos['lon'], mouse_pos['lat'])

    ax.coastlines(resolution='110m', linewidth=1)
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False,
        linewidth=0.5, xlocs=np.arange(-180,181,1),
        ylocs=np.arange(-90,91,1)
    )

    if show_equator:
        lons = np.linspace(-180, 180, 361)
        ax.plot(
            lons, np.zeros_like(lons),
            transform=ccrs.PlateCarree(),
            color='red', linestyle='--', linewidth=2
        )

    plt.title(
        f"Center: lon={center['lon']:.1f}°, lat={center['lat']:.1f}°   "
        "Keys: ←↑→↓ rotate • E equator • I imagery • P peat patch • "
        "C cell area • R resolution",
        fontsize=9
    )

# -----------------------------------------------------------------------------
# Set up figure & callbacks
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(8, 8))
ax  = plt.axes(projection=ccrs.Orthographic())
draw_globe()

def on_move(event):
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        mouse_pos['lon'] = mouse_pos['lat'] = None
    else:
        mouse_pos['lon'], mouse_pos['lat'] = ccrs.PlateCarree().transform_point(
            x, y, src_crs=ax.projection
        )
    if show_peatland:
        draw_globe()
        fig.canvas.draw_idle()

def on_key(event):
    global show_equator, show_imagery, show_peatland
    global lat_step, lon_step

    k = event.key.lower()
    if k == 'left':
        center['lon'] -= 10
    elif k == 'right':
        center['lon'] += 10
    elif k == 'up':
        center['lat'] = min(center['lat'] + 10, 90)
    elif k == 'down':
        center['lat'] = max(center['lat'] - 10, -90)
    elif k == 'e':
        show_equator = not show_equator
    elif k == 'i':
        show_imagery = not show_imagery
    elif k == 'p':
        show_peatland = not show_peatland
    elif k == 'c':
        lon, lat = mouse_pos['lon'], mouse_pos['lat']
        if lon is None:
            print("Move cursor over the globe first!")
        else:
            lat1 = np.floor(lat/lat_step)*lat_step
            lat2 = lat1 + lat_step
            lon1 = np.floor(lon/lon_step)*lon_step
            lon2 = lon1 + lon_step
            φ1, φ2 = np.radians(lat1), np.radians(lat2)
            area = R**2 * abs(np.radians(lon2-lon1)*(np.sin(φ2)-np.sin(φ1)))
            print(f"\nCursor lat={lat:.4f}°, lon={lon:.4f}° → {area:.2f} km²\n")
    elif k == 'r':
        try:
            lat_step = float(input("New latitude Δ°: "))
            lon_step = float(input("New longitude Δ°: "))
        except ValueError:
            print("Need numeric values.")
    draw_globe()
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('key_press_event',     on_key)

plt.show()
