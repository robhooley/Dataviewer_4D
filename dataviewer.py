import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from natsort import natsorted
import rsciio.blockfile as rs_blo
import rsciio.hspy      as rs_hspy
from PIL import Image, ImageTk, ImageDraw,ImageFont
import os,inspect
from analysis_functions import *
import tifffile as tiff
import threading,queue
import numpy as np
import json

# Global variable to control mouse motion functionality
mouse_motion_enabled = True  # Initially enabled

def ask_scan_width(parent, guessed_width: int, num_files: int) -> int | None:
    """Modal integer prompt with initial value and proper parenting."""
    return simpledialog.askinteger(
        title="Enter Scan Width",
        prompt=f"Detected {num_files} images.\nGuessed width: {guessed_width}\nEnter scan width (X):",
        initialvalue=int(guessed_width),
        minvalue=1,
        maxvalue=num_files,
        parent=parent
    )
def ask_pixel_size_nm(parent, guessed_nm: float | None = None) -> float | None:
    return simpledialog.askfloat(
        title="Pixel size (nm/pixel)",
        prompt="Enter pixel size in nanometers (nm/pixel):",
        initialvalue=None if guessed_nm is None else float(guessed_nm),
        minvalue=0.0,
        parent=parent
    )

def _normalize_navigator(img2d: np.ndarray) -> np.ndarray:
    a = np.asarray(img2d, dtype=np.float32)
    mask = np.isfinite(a)
    vals = a[mask]
    if vals.size == 0:
        return np.zeros_like(a, dtype=np.uint8)
    lo, hi = np.percentile(vals, (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return np.zeros_like(a, dtype=np.uint8)
    a = np.clip(a, lo, hi)
    den = (a.max() - a.min()) or 1.0
    return ((a - a.min()) / den * 255).astype(np.uint8)

def downsample_diffraction(array4d, rescale_to=128, mode='sum'):
    if array4d.ndim != 4:
        raise ValueError("Expected a 4-D array (scan_y, scan_x, dp_y, dp_x)")
    ny_s, nx_s, ny_dp, nx_dp = array4d.shape
    if ny_dp % rescale_to or nx_dp % rescale_to:
        raise ValueError(f"DP size must be a multiple of {rescale_to}; got ({ny_dp}, {nx_dp}).")
    fy, fx = ny_dp // rescale_to, nx_dp // rescale_to
    if fy != fx:
        raise ValueError("Diffraction pattern must be square.")
    a = np.ascontiguousarray(array4d)
    v = a.reshape(ny_s, nx_s, rescale_to, fy, rescale_to, fx)
    if mode == 'sum':
        return v.sum(axis=(-1, -3))
    elif mode == 'mean':
        return v.mean(axis=(-1, -3))
    raise ValueError("mode must be 'sum' or 'mean'")

def array2blo(data: np.ndarray, px_nm: float, save_path: str,
              intensity_scaling="crop", bin_data_to_128=True):
    """Save (Ny, Nx, Dy, Dx) array to .blo via rsciio with minimal metadata."""


    # Optional binning to 128×128 DPs
    if bin_data_to_128 and (data.shape[2] != 128 or data.shape[3] != 128):
        data = downsample_diffraction(data, rescale_to=128, mode="sum")

    if data.ndim != 4:
        raise ValueError("`data` must be 4-D (Ny, Nx, Dy, Dx)")
    Ny, Nx, Dy, Dx = map(int, data.shape)

    # Navigator image (for preview in .blo viewers)
    navigator = _normalize_navigator(data.sum(axis=(2, 3)))

    # Minimal metadata
    meta = {"Pixel size (nm)": float(px_nm)}
    axes = [
        {"name": "scan_y", "units": "nm", "index_in_array": 0,
         "size": Ny, "offset": 0.0, "scale": px_nm, "navigate": True},
        {"name": "scan_x", "units": "nm", "index_in_array": 1,
         "size": Nx, "offset": 0.0, "scale": px_nm, "navigate": True},
        {"name": "qy", "units": "px", "index_in_array": 2,
         "size": Dy, "offset": 0.0, "scale": 1.0, "navigate": False},
        {"name": "qx", "units": "px", "index_in_array": 3,
         "size": Dx, "offset": 0.0, "scale": 1.0, "navigate": False},
    ]

    signal = {
        "data": data,
        "axes": axes,
        "metadata": {"acquisition": meta},
        "original_metadata": meta,
        "attributes": {"_lazy": False},
    }

    rs_blo.file_writer(
        save_path,
        signal,
        intensity_scaling=intensity_scaling,
        navigator=navigator,
        endianess="<",
        show_progressbar=True,
    )

# Stores how the current PIL image is placed on the canvas (for mapping clicks)
display_geom = {
    "img_w": 0, "img_h": 0,     # resized image size (px)
    "x0": 0, "y0": 0,           # top-left corner of image on canvas (px)
    "scale_x": 1.0, "scale_y": 1.0,  # image pixels per canvas pixel (inverse of resize factor)
}

def _resize_preserve_aspect(img_pil, max_side=1024):
    w, h = img_pil.size
    if w == 0 or h == 0:
        return img_pil
    if w >= h:
        new_w = max_side
        new_h = max(1, int(round(h * (max_side / w))))
    else:
        new_h = max_side
        new_w = max(1, int(round(w * (max_side / h))))
    return img_pil.resize((new_w, new_h), Image.NEAREST)




def visualiser():
    """
    Creates a Tkinter application for visualising a 4D STEM array with import, export and saving functionality.
    """
    root = tk.Tk()
    root.title("4D Array Visualiser")
    root.grid_rowconfigure(0, weight=1)  # canvases
    root.grid_columnconfigure(0, weight=1)  # left panel
    root.grid_columnconfigure(1, weight=0)  # right panel


    # Global variables to store the data and selected function
    data_array = None
    selected_function = tk.StringVar(value="BF")
    radius_value = tk.IntVar(value=25)  # Default radius value
    circle_center = [None, None]  # Default circle center (to be dynamically set)
    current_scan_xy = [-1, -1]  # 

    # --- Diffraction display adjustments (right panel only) ---
    dp_gamma = tk.DoubleVar(value=1.0)       # 0.2–5.0
    dp_brightness = tk.DoubleVar(value=0.0)  # -100..100
    dp_contrast = tk.DoubleVar(value=1.0)    # 0.2..3.0
    dp_log_intensity = tk.BooleanVar(value=False)

    def apply_bcg_u8(img_u8: np.ndarray,
                     gamma: float,
                     brightness: float,
                     contrast: float) -> np.ndarray:
        """Apply brightness/contrast/gamma to an 8-bit grayscale image (uint8)."""
        if img_u8.dtype != np.uint8:
            img_u8 = img_u8.astype(np.uint8, copy=False)

        g = float(gamma) if gamma is not None else 1.0
        g = max(0.01, g)

        c = float(contrast) if contrast is not None else 1.0
        c = max(0.0, c)

        b = float(brightness) if brightness is not None else 0.0
        # Map (-100..100) → approx (-255..255)
        b = (b / 100.0) * 255.0

        x = img_u8.astype(np.float32)

        # Contrast around mid-gray then brightness
        x = (x - 128.0) * c + 128.0 + b
        x = np.clip(x, 0.0, 255.0)

        # Gamma: use 1/g so gamma>1 brightens midtones (more intuitive here)
        x = 255.0 * np.power(x / 255.0, 1.0 / g)

        return np.clip(x, 0.0, 255.0).astype(np.uint8)

    #'[x, y] of the scan position currently shown on the right
    df_centers = [list(circle_center)]  # always keep at least one DF center
    # Canvas and state variables
    main_image_pil = None
    pointer_image_pil = None
    clicked_positions = []  # Store positions of clicks
    image_refs = {"main_image": None, "current_sub_image": None}  # Prevent garbage collection


    functions = {
        "BF": {"fn": VBF, "params": ["radius", "center"]},
        "DF": {"fn": VDF_multi, "params": ["radius", "centers"]},  # handled specially
        "VADF": {"fn": VADF, "params": ["inner_radius", "outer_radius", "center"]},
        "Cross Correlation": {"fn": cross_correlation_map, "params":None}
    }

    # ---- Metadata state ----
    current_metadata = {}  # last loaded metadata dict

    def _metadata_clear():
        current_metadata.clear()
        for i in metadata_tree.get_children():
            metadata_tree.delete(i)

    def _insert_tree_dict(tree, parent, data):
        """Recursively insert dict/list/values into a Treeview."""
        if isinstance(data, dict):
            for k, v in data.items():
                node = tree.insert(parent, "end", text=str(k), values=("",))
                _insert_tree_dict(tree, node, v)
        elif isinstance(data, (list, tuple)):
            for i, v in enumerate(data):
                node = tree.insert(parent, "end", text=f"[{i}]", values=("",))
                _insert_tree_dict(tree, node, v)
        else:
            # leaf value -> put it in 'Value' column; keep node label in 'Key' column
            s = repr(data)
            if len(s) > 500:  # avoid huge blobs
                s = s[:497] + "..."
            tree.insert(parent, "end", text="", values=(s,))

    def _metadata_set(meta: dict | None):
        """Replace the metadata view with a new dict."""
        _metadata_clear()
        if not meta:
            return
        current_metadata.update(meta)
        # Add a single root node for readability
        root_id = metadata_tree.insert("", "end", text="metadata", values=("",))
        _insert_tree_dict(metadata_tree, root_id, meta)
        metadata_tree.item(root_id, open=True)

    def load_metadata_json():
        """User picks a JSON file; display it in the metadata panel."""
        path = filedialog.askopenfilename(parent=root, filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            messagebox.showerror("Load metadata (.json)", f"Failed to read JSON:\n{e}", parent=root)
            return
        _metadata_set(meta)

    def clamp_radius(event=None):
        """
        Snap the radius to [1, 256] when user presses Enter or leaves the field.
        If the field is empty or non-numeric, fall back to the last valid value.
        """
        # last known-good value
        current = radius_value.get()
        s = radius_entry.get().strip()

        if s == "":
            val = current
        else:
            try:
                val = int(s)
            except ValueError:
                val = current

        # clamp
        val = 1 if val < 1 else (256 if val > 256 else val)

        # reflect in UI + state
        radius_value.set(val)
        radius_entry.delete(0, tk.END)
        radius_entry.insert(0, str(val))

        # re-run your computation if radius affects it
        update_function()

    def save_images():
        """
        Saves the left image (main image with markers), sub-images for clicked positions,
        and the list of clicked positions to disk.
        """
        if data_array is None or main_image_pil is None:
            print("No data or main image available to save.")
            return

        # Ask user to select a directory
        save_dir = filedialog.askdirectory(parent=root)
        if not save_dir:
            print("Save canceled.")
            return

        # Save the main image with markers
        main_image_with_markers = main_image_pil.copy()
        draw = ImageDraw.Draw(main_image_with_markers)
        for i, (x, y) in enumerate(clicked_positions, start=1):
            # Draw markers on the main image
            draw.line((x - 3, y, x + 3, y), fill="red", width=1)
            draw.line((x, y - 3, x, y + 3), fill="red", width=1)
            draw.text((x + 5, y - 5), str(i), fill="red")

        # Save the main image
        main_image_path = f"{save_dir}/main_image_with_markers.tiff"
        main_image_with_markers.save(main_image_path)

        # Save individual sub-images
        for i, (x, y) in enumerate(clicked_positions, start=1):
            if 0 <= y < data_array.shape[0] and 0 <= x < data_array.shape[1]:
                sub_image = data_array[y, x]
                sub_image_path = f"{save_dir}/clicked_position_{i}_sub_image.tiff"
                Image.fromarray(sub_image).save(sub_image_path)

        # Save clicked positions to a text file
        positions_path = f"{save_dir}/clicked_positions.txt"
        with open(positions_path, "w") as f:
            for i, (x, y) in enumerate(clicked_positions, start=1):
                f.write(f"Position {i}: ({x}, {y})\n")

        print(f"Images and positions saved to {save_dir}")
        messagebox.showinfo("Save Complete", f"Images and positions saved to {save_dir}")

    def load_series():
        """Threaded TIFF series loader with progress bar + Cancel."""
        nonlocal data_array, circle_center, df_centers

        folder_path = filedialog.askdirectory(mustexist=True, parent=root)
        if not folder_path:
            return

        files = [f for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f))
                 and f.lower().endswith(('.tif', '.tiff'))]
        if not files:
            messagebox.showwarning("Load TIFF Series", "No .tif/.tiff files found.", parent=root)
            return

        try:
            files = natsorted(files)
        except Exception:
            files = sorted(files)
        N = len(files)

        guessed_scan_width = int(np.sqrt(N))
        scan_width = ask_scan_width(root, guessed_scan_width, N)
        if not scan_width:
            return
        if N % scan_width != 0:
            messagebox.showwarning("Load TIFF Series",
                                   f"{N} not divisible by {scan_width}; last row will be padded.",
                                   parent=root)
        scan_height = (N + (scan_width - 1)) // scan_width  # ceil

        # Progress UI (embedded under canvases)
        progress_frame = tk.Frame(root, relief="sunken", borderwidth=1)
        progress_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=2)

        progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate", maximum=N)
        progress_bar.grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        progress_frame.grid_columnconfigure(0, weight=1)

        progress_label = tk.Label(progress_frame, text=f"0 / {N} (0.0%)")
        progress_label.grid(row=0, column=1, padx=6)

        cancel_flag = {'stop': False}
        cancel_btn = tk.Button(progress_frame, text="Cancel", command=lambda: cancel_flag.update(stop=True))
        cancel_btn.grid(row=0, column=2, padx=6)

        q = queue.Queue()

        def _read_one(p):
            try:
                return tiff.imread(p)
            except Exception:
                with Image.open(p) as im:
                    return np.array(im)

        def worker():
            try:
                first = _read_one(os.path.join(folder_path, files[0]))
                if first.ndim != 2:
                    q.put(('error', f"Expected 2D frames, got {first.shape}"))
                    return
                H, W = first.shape
                dtype = first.dtype
                stack = np.empty((N, H, W), dtype=dtype)
                stack[0] = first
                q.put(('progress', 1, N))

                for i, name in enumerate(files[1:], start=2):
                    if cancel_flag['stop']:
                        q.put(('cancelled',))
                        return
                    arr = _read_one(os.path.join(folder_path, name))
                    if arr.shape != (H, W):
                        q.put(('error', f"Frame {name} shape {arr.shape} != {(H, W)}"))
                        return
                    stack[i - 1] = arr
                    q.put(('progress', i, N))

                if N % scan_width != 0:
                    missing = scan_width - (N % scan_width)
                    pad = np.zeros((missing, H, W), dtype=dtype)
                    stack = np.concatenate([stack, pad], axis=0)
                data = stack.reshape(-1, scan_width, H, W)  # rows auto (ceil)
                q.put(('done', data))
            except Exception as e:
                q.put(('error', str(e)))

        def pump():
            nonlocal data_array, circle_center, df_centers
            try:
                while True:
                    msg = q.get_nowait()
                    kind = msg[0]
                    if kind == 'progress':
                        i, total = msg[1], msg[2]
                        progress_bar['value'] = i
                        progress_label.config(text=f"{i} / {total} ({(i / total) * 100:.1f}%)")
                    elif kind == 'done':
                        data_array = msg[1]
                        H, W = data_array.shape[2], data_array.shape[3]
                        circle_center[:] = [W // 2, H // 2]
                        df_centers[:] = [list(circle_center)]
                        #update_pointer_image(W // 2, H // 2)
                        current_scan_xy[:] = [data_array.shape[1] // 2, data_array.shape[0] // 2]


                        progress_frame.destroy()

                        update_main_image(trigger_user_function=True)
                        update_pointer_image(*current_scan_xy)
                        return
                    elif kind == 'cancelled':
                        progress_frame.destroy()
                        messagebox.showinfo("Load TIFF Series", "Cancelled.", parent=root)
                        return
                    elif kind == 'error':
                        progress_frame.destroy()
                        messagebox.showerror("Load TIFF Series", msg[1], parent=root)
                        return
            except queue.Empty:
                pass
            root.after(33, pump)

        threading.Thread(target=worker, daemon=True).start()
        pump()

    def resize_image(image, max_dimension):
        """
        Resizes an image to fit within the specified maximum dimension, preserving aspect ratio.
        Uses nearest-neighbor interpolation to keep the image pixelated.
        """
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            # Fit by width
            new_width = max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            # Fit by height
            new_height = max_dimension
            new_width = int(new_height / aspect_ratio)

        return image.resize((new_width, new_height), Image.NEAREST)

    def ensure_rgb(image):
        """
        Ensures the image is in RGB format. If grayscale, converts it to RGB.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def normalize_to_8bit(img, clip=(0.5, 99.5), ignore_zeros=True, use_log=False):
        a = np.asarray(img).astype(np.float32)
        mask = np.isfinite(a)
        if ignore_zeros: mask &= (a != 0)
        if not mask.any(): return np.zeros(a.shape, np.uint8)
        vals = a[mask]
        lo, hi = np.percentile(vals, clip)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: lo, hi = vals.min(), vals.max()
        if hi <= lo: v = lo; eps = max(abs(v) * 1e-3, 1e-6); lo, hi = v - eps, v + eps
        a = np.clip(a, lo, hi)
        if use_log:
            if lo <= 0: a += -(lo) + 1e-6
            a = np.log1p(a)
        denom = (a.max() - a.min()) or 1.0
        out = ((a - a.min()) / denom * 255).astype(np.uint8)

        return out

    def run_with_spinner(root, title, work_fn, done_cb):
        win = tk.Toplevel(root)
        win.title(title)
        win.transient(root)
        win.grab_set()
        lbl = tk.Label(win, text="Working…")
        lbl.pack(padx=20, pady=10)
        bar = ttk.Progressbar(win, mode="indeterminate", length=200)
        bar.pack(padx=20, pady=10)
        bar.start(20)

        def worker():
            try:
                result = work_fn()
                root.after(0, lambda: finish(result, None))
            except Exception as e:
                root.after(0, lambda: finish(None, e))

        def finish(result, error):
            bar.stop()
            win.grab_release()
            win.destroy()
            if error:
                messagebox.showerror(title, str(error), parent=root)
            else:
                done_cb(result)

        threading.Thread(target=worker, daemon=True).start()

    def update_main_image(trigger_user_function: bool = False):
        """
        Updates the main (navigator) image in the left canvas.

        If `trigger_user_function` is True, it regenerates the main image using
        the currently selected analysis function. Otherwise, it just redraws
        whatever is in `main_image_pil`.
        """
        nonlocal main_image_pil

        # Optionally regenerate from analysis function
        if trigger_user_function:
            if data_array is None:
                print("No data array available.")
                return
            sel = selected_function.get()
            print(f"update_main_image(): regenerating via {sel}")
            try:
                if sel == "DF":
                    arr = VDF_multi(data_array, radius_value.get(), df_centers)
                elif sel == "VADF":
                    outer_radius = max(radius_value.get() + 1, outer_radius_value.get())
                    arr = functions[sel]["fn"](data_array,
                                               radius_value.get(),
                                               outer_radius,
                                               tuple(circle_center))
                else:
                    arr = functions[sel]["fn"](data_array,
                                               radius_value.get(),
                                               tuple(circle_center))
                normalized = normalize_to_8bit(arr)
                main_image_pil = Image.fromarray(normalized).convert("RGB")
                print(f"update_main_image(): new image shape {arr.shape}")
            except Exception as e:
                messagebox.showerror("Analysis Error", str(e), parent=root)
                return

        if main_image_pil is None:
            main_canvas.delete("all")
            print("update_main_image(): no image to draw")
            return

        # --- Resize to fit canvas while preserving aspect ---
        iw0, ih0 = main_image_pil.size
        cw = max(1, main_canvas.winfo_width())
        ch = max(1, main_canvas.winfo_height())

        if iw0 >= ih0:
            iw = min(cw, 1024)
            ih = max(1, int(round(ih0 * (iw / iw0))))
        else:
            ih = min(ch, 1024)
            iw = max(1, int(round(iw0 * (ih / ih0))))

        resized_main = main_image_pil.resize((iw, ih), Image.NEAREST)
        resized_main = ensure_rgb(resized_main)

        # --- Center it in canvas ---
        x0 = max(0, (cw - iw) // 2)
        y0 = max(0, (ch - ih) // 2)

        # --- Draw markers ---
        draw = ImageDraw.Draw(resized_main)
        sx = iw / iw0 if iw0 else 1.0
        sy = ih / ih0 if ih0 else 1.0
        ImageDraw.ImageDraw.font = ImageFont.truetype("arial.ttf")
        for i, (x, y) in enumerate(clicked_positions, start=1):
            rx = int(round((x + 0.5) * sx))
            ry = int(round((y + 0.5) * sy))
            draw.line((rx - 3, ry, rx + 3, ry), fill="red", width=1)
            draw.line((rx, ry - 3, rx, ry + 3), fill="red", width=1)
            draw.text((rx + 5, ry - 5), str(i), fill="red",font_size=28)

        # --- Push to canvas ---
        main_image_tk = ImageTk.PhotoImage(resized_main)
        image_refs["main_image"] = main_image_tk
        main_canvas.delete("all")
        main_canvas.create_image(x0, y0, anchor=tk.NW, image=main_image_tk)

        # Save display geometry for click/motion mapping
        main_canvas.display_geom = {
            "img_w": iw, "img_h": ih,
            "x0": x0, "y0": y0,
            "orig_w": iw0, "orig_h": ih0,
            "scale_x": (iw0 / iw) if iw else 1.0,
            "scale_y": (ih0 / ih) if ih else 1.0,
        }
        #print("update_main_image(): finished drawing")

    def remove_last_marker():
        """
        Removes the last clicked marker from the navigator and refreshes the view.
        Also ensures it won't be exported in images/positions.
        """
        if not clicked_positions:
            messagebox.showinfo("Remove Marker", "No markers to remove.", parent=root)
            return

        # Remove last position
        removed = clicked_positions.pop()
        print(f"Removed marker at {removed}")

        # Redraw the main image without that marker
        update_main_image()

    def clear_all_markers():
        "Remove all "
        if not clicked_positions:
            messagebox.showinfo("Clear all markers", "No markers to remove.", parent=root)
            return
        clicked_positions.clear()
        update_main_image()

    def update_pointer_image(x, y):
        """
        Update the right (pointer) canvas with the diffraction pattern at (x, y).
        Draws detector overlays:
          - DF  : multiple green circles (one per center) or a hint if none.
          - VADF: two concentric red circles (inner & outer).
          - BF/ADF/others: single red circle.
        """
        nonlocal pointer_image_pil
        if data_array is None or main_image_pil is None:
            return

        #print("Pointer update:", x, y, "| data_array shape:", data_array.shape)

        # --- Clamp indices and sync global scan position ---
        Y, X = data_array.shape[:2]
        x = max(0, min(X - 1, int(x)))
        y = max(0, min(Y - 1, int(y)))
        current_scan_xy[:] = [x, y]

        # --- Extract + normalize diffraction pattern ---
        sub_image = data_array[y, x]

        # Normalize (optionally log-scale) to 8-bit for display
        normalized = normalize_to_8bit(sub_image, use_log=dp_log_intensity.get())

        # Apply display adjustments (right panel only)
        adjusted = apply_bcg_u8(
            normalized,
            gamma=dp_gamma.get(),
            brightness=dp_brightness.get(),
            contrast=dp_contrast.get()
        )

        pointer_image_pil = Image.fromarray(adjusted).convert("RGB")

        # --- Scale to display size ---
        resized = pointer_image_pil.resize((SQUARE_CANVAS_SIZE, SQUARE_CANVAS_SIZE), Image.NEAREST)
        draw = ImageDraw.Draw(resized, "RGBA")

        # --- Common scaling factors for overlay geometry ---
        scale_x = SQUARE_CANVAS_SIZE / pointer_image_pil.width
        scale_y = SQUARE_CANVAS_SIZE / pointer_image_pil.height

        sel = selected_function.get()

        if sel == "DF":
            r = int(radius_value.get())
            if df_centers:
                for i, (cx, cy) in enumerate(df_centers):
                    ux, uy = int(cx * scale_x), int(cy * scale_y)
                    rx, ry = r * scale_x, r * scale_y
                    draw.ellipse((ux - rx, uy - ry, ux + rx, uy + ry),
                                 outline=(0, 255, 0, 128), width=2)
                    draw.line((ux - 3, uy, ux + 3, uy), fill="red", width=1)
                    draw.line((ux, uy - 3, ux, uy + 3), fill="red", width=1)
                    draw.text((ux + 5, uy - 5), f"{i + 1}", fill="red")
            else:
                # no centers selected → show a small hint
                draw.text((8, 8), "No DF centers", fill="red")

        elif sel == "VADF":
            # Two concentric circles: inner (radius_value), outer (outer_radius_value)
            rin = int(radius_value.get())
            rout = int(outer_radius_value.get())
            if rout < rin:
                rout = rin  # keep sensible ordering visually

            cx, cy = circle_center
            ux, uy = int(cx * scale_x), int(cy * scale_y)
            rinx, riny = rin * scale_x, rin * scale_y
            routx, routy = rout * scale_x, rout * scale_y

            # Inner ring
            draw.ellipse((ux - rinx, uy - riny, ux + rinx, uy + riny),
                         outline=(255, 0, 0, 160), width=3)
            # Outer ring
            draw.ellipse((ux - routx, uy - routy, ux + routx, uy + routy),
                         outline=(255, 0, 0, 90), width=3)

            # Center crosshair
            draw.line((ux - 3, uy, ux + 3, uy), fill="red", width=1)
            draw.line((ux, uy - 3, ux, uy + 3), fill="red", width=1)

        else:
            # Default single-circle overlay (BF / ADF / others)
            r = int(radius_value.get())
            cx, cy = circle_center
            ux, uy = int(cx * scale_x), int(cy * scale_y)
            rx, ry = r * scale_x, r * scale_y
            draw.ellipse((ux - rx, uy - ry, ux + rx, uy + ry),
                         outline=(255, 0, 0, 128), width=3)
            draw.line((ux - 3, uy, ux + 3, uy), fill="red", width=1)
            draw.line((ux, uy - 3, ux, uy + 3), fill="red", width=1)

        # --- Push to Tkinter canvas ---
        pointer_image_tk = ImageTk.PhotoImage(resized)
        image_refs["current_sub_image"] = pointer_image_tk
        pointer_canvas.delete("all")
        pointer_canvas.create_image(0, 0, anchor=tk.NW, image=pointer_image_tk)

    def on_pointer_click(event):
        """
        Handle clicks on the diffraction (right) canvas.
        - DF mode: left click adds a center, right click removes last (can reach zero).
        - Other modes: left/right click set the single circle center.
        """
        nonlocal df_centers, circle_center
        if pointer_image_pil is None:
            return

        dw, dh = SQUARE_CANVAS_SIZE, SQUARE_CANVAS_SIZE
        if not (0 <= event.x < dw and 0 <= event.y < dh):
            return

        # Map canvas coords back to original DP coords
        ow, oh = pointer_image_pil.width, pointer_image_pil.height
        cx = int(event.x * (ow / dw))
        cy = int(event.y * (oh / dh))

        if selected_function.get() == "DF":
            if event.num == 1:  # left click → add DF center
                df_centers.append([cx, cy])
            elif event.num == 3:  # right click → remove last DF center
                if df_centers:
                    df_centers.pop()

        else:
            # For BF / VADF / etc.: always set a single center
            circle_center[0], circle_center[1] = cx, cy

        update_pointer_image(current_scan_xy[0], current_scan_xy[1])
        update_function()

    def on_click(event):
        """
        Add a marker by clicking on the left (navigator) image.
        Uses geometry recorded by update_main_image() to map to navigator pixels.
        """
        if main_image_pil is None:
            return
        geom = getattr(main_canvas, "display_geom", None)
        if not geom:
            return

        x0, y0 = geom["x0"], geom["y0"]
        iw, ih = geom["img_w"], geom["img_h"]
        if not (x0 <= event.x < x0 + iw and y0 <= event.y < y0 + ih):
            return

        rx = event.x - x0
        ry = event.y - y0
        x_nav = int(rx * geom["scale_x"])
        y_nav = int(ry * geom["scale_y"])

        ow, oh = geom["orig_w"], geom["orig_h"]
        x_nav = max(0, min(ow - 1, x_nav))
        y_nav = max(0, min(oh - 1, y_nav))

        clicked_positions.append((x_nav, y_nav))
        current_scan_xy[0], current_scan_xy[1] = x_nav, y_nav  # <<< keep in sync
        update_pointer_image(x_nav, y_nav)  # <<< reflect immediately
        update_main_image()

    def toggle_mouse_motion(event=None):
        """
        Toggle mouse motion on/off with middle click.
        """
        global mouse_motion_enabled
        mouse_motion_enabled = not mouse_motion_enabled
        state = "on" if mouse_motion_enabled else "off"
        #print(f"Mouse motion functionality is now {state}.")
        try:
            # keep last coords if any; just reflect state
            txt = left_status_var.get()
            # overwrite the tail after ' | '
            if " | " in txt:
                left_status_var.set(txt.split(" | ")[0] + f"  |  motion: {state}")
            else:
                left_status_var.set(f"Mouse: (—, —)  |  motion: {state}")
        except Exception:
            pass

    def _on_mouse_motion(event):
        """
        Updates the pointer image & labels based on cursor position
        over the navigator (left) image.
        """
        global mouse_motion_enabled
        if not mouse_motion_enabled:  # paused: do nothing
            return
        if data_array is None or main_image_pil is None:
            return

        geom = getattr(main_canvas, "display_geom", None)
        if not geom:
            return

        x0, y0 = geom["x0"], geom["y0"]
        iw, ih = geom["img_w"], geom["img_h"]
        if not (x0 <= event.x < x0 + iw and y0 <= event.y < y0 + ih):
            return

        rx = event.x - x0
        ry = event.y - y0
        x_nav = int(rx * geom["scale_x"])
        y_nav = int(ry * geom["scale_y"])

        ow, oh = geom["orig_w"], geom["orig_h"]
        x_nav = max(0, min(ow - 1, x_nav))
        y_nav = max(0, min(oh - 1, y_nav))

        # Update state + redraw pointer image
        current_scan_xy[0], current_scan_xy[1] = x_nav, y_nav
        update_pointer_image(x_nav, y_nav)

        # Update labels
        try:
            left_status_var.set(f"Mouse: ({x_nav}, {y_nav})  |  motion: on")
            # Show scan index and current detector center
            right_status_var.set(f"Scan: ({x_nav}, {y_nav})  |  center: ({circle_center[0]}, {circle_center[1]})")
        except Exception:
            pass

    def load_npy_with_progress(path):
        mm = np.load(path, mmap_mode='r')
        out = np.empty_like(mm)
        total = mm.size
        chunk = max(1, total // 200)  # ~200 steps

        # Embedded progress frame
        progress_frame = tk.Frame(root, relief="sunken", borderwidth=1)
        progress_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=2)

        bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate", maximum=100)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        progress_frame.grid_columnconfigure(0, weight=1)

        lbl = tk.Label(progress_frame, text="0%")
        lbl.grid(row=0, column=1, padx=8)

        # Actual copy loop
        flat_src = mm.ravel()
        flat_dst = out.ravel()
        for start in range(0, total, chunk):
            end = min(total, start + chunk)
            flat_dst[start:end] = flat_src[start:end]
            pct = (end / total) * 100
            bar["value"] = pct
            lbl.config(text=f"{pct:.1f}%")
            root.update_idletasks()

        progress_frame.destroy()
        return out

    def _load_with_progress(start_fn, title: str):
        """
        start_fn(cancel_flag: dict, q: queue.Queue) should push:
            ('status', 'message')    optional, update status label
            ('done', ndarray)        on success
            ('error', 'message')     on failure
        """
        progress_win = tk.Toplevel(root)
        progress_win.title(title)
        progress_win.transient(root)
        progress_win.grab_set()  # modal dialog

        tk.Label(progress_win, text=title).pack(padx=12, pady=(12, 4))

        bar = ttk.Progressbar(progress_win, orient="horizontal", length=340, mode="indeterminate")
        bar.pack(padx=12, pady=8)
        bar.start(10)  # ms per step

        status_lbl = tk.Label(progress_win, text="Starting…")
        status_lbl.pack(padx=12, pady=(0, 10))

        btn_frame = tk.Frame(progress_win);
        btn_frame.pack(pady=(0, 12))
        cancel_flag = {'stop': False}

        def _cancel():
            cancel_flag['stop'] = True
            status_lbl.config(text="Cancelling…")

        tk.Button(btn_frame, text="Cancel", command=_cancel).pack()

        q = queue.Queue()

        def worker():
            try:
                start_fn(cancel_flag, q)
            except Exception as e:
                q.put(('error', f"{type(e).__name__}: {e}"))

        def pump():
            def _finish_with_array_and_meta(arr, meta):
                nonlocal data_array, circle_center,df_centers
                arr4d = _coerce_to_4d(arr, root, ask_scan_width)
                if arr4d is None:
                    bar.stop();
                    progress_win.grab_release();
                    progress_win.destroy()
                    return
                data_array = arr4d
                H, W = data_array.shape[2], data_array.shape[3]
                circle_center[:] = [W // 2, H // 2]
                df_centers = [list(circle_center)]
                update_pointer_image(data_array.shape[1] // 2, data_array.shape[0] // 2)

                if isinstance(meta, dict):
                    _metadata_set(meta)
                bar.stop();
                progress_win.grab_release();
                progress_win.destroy()
                messagebox.showinfo(title, f"Loaded → navigator {data_array.shape[:2]}", parent=root)
                update_function()

            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == 'status':
                        status_lbl.config(text=str(payload))
                    elif kind == 'done':
                        if cancel_flag['stop']:
                            bar.stop();
                            progress_win.grab_release();
                            progress_win.destroy()
                            messagebox.showinfo(title, "Cancelled.", parent=root)
                            return
                        # payload may be ndarray or (ndarray, meta)
                        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], np.ndarray):
                            _finish_with_array_and_meta(payload[0], payload[1])
                        elif isinstance(payload, np.ndarray):
                            _finish_with_array_and_meta(payload, {})
                        else:
                            bar.stop();
                            progress_win.grab_release();
                            progress_win.destroy()
                            messagebox.showerror(title, "Unexpected loader payload.", parent=root)
                        return
                    elif kind == 'error':
                        bar.stop();
                        progress_win.grab_release();
                        progress_win.destroy()
                        messagebox.showerror(title, str(payload), parent=root)
                        return
            except queue.Empty:
                pass
            root.after(50, pump)

        threading.Thread(target=worker, daemon=True).start()
        pump()

    def _coerce_to_4d(arr, parent, ask_width_fn):
        arr = np.asarray(arr)
        if arr.ndim == 4:
            return arr
        if arr.ndim == 3:
            N, H, W = arr.shape
            guessed = int(np.sqrt(N))
            scan_width = ask_width_fn(parent, guessed, N)
            if not scan_width:
                return None
            if N % scan_width != 0:
                missing = scan_width - (N % scan_width)
                arr = np.concatenate([arr, np.zeros((missing, H, W), dtype=arr.dtype)], axis=0)
                N = arr.shape[0]
            return arr.reshape(N // scan_width, scan_width, H, W)
        messagebox.showerror("File Load", f"Expected 3D or 4D array, got {arr.shape}", parent=parent)
        return None

    def _pick_dataset_index(parent, datasets) -> int | None:
        """Ask the user which dataset to load if there are multiple."""
        labels = []
        for i, d in enumerate(datasets):
            data = d.get("data")
            shape = getattr(data, "shape", None)
            dtype = getattr(data, "dtype", None)
            labels.append(f"{i}: shape={shape}, dtype={dtype}")
        msg = "Multiple datasets found:\n" + "\n".join(labels) + "\n\nEnter index:"
        return simpledialog.askinteger("Select dataset", msg, parent=parent,
                                       minvalue=0, maxvalue=len(datasets) - 1)

    def _read_blo_rsciio_interactive(path: str, parent=None, lazy: bool = False, endianess: str = "<"):
        """Read BLO via rsciio; if multiple datasets exist, ask the user which to load.
           Returns (np.ndarray, metadata_dict) or (None, None) if user cancels.
        """
        path = os.path.normpath(str(path))
        sig = inspect.signature(rs_blo.file_reader)
        kwargs = {}
        if "lazy" in sig.parameters:      kwargs["lazy"] = lazy
        if "endianess" in sig.parameters: kwargs["endianess"] = endianess
        datasets = rs_blo.file_reader(path, **kwargs)

        if not isinstance(datasets, (list, tuple)) or not datasets:
            raise ValueError("BLO reader returned no datasets.")

        # Choose dataset (auto if single, prompt if multiple)
        if len(datasets) == 1:
            d = datasets[0]
        else:
            idx = _pick_dataset_index(parent, datasets)
            if idx is None:
                return None, None  # user cancelled
            d = datasets[idx]

        data = d.get("data")
        if data is None:
            raise ValueError("Selected dataset has no 'data' field.")

        # Materialize to RAM
        if hasattr(data, "compute"):  # dask → numpy
            data = data.compute()
        if isinstance(data, np.memmap):  # memmap → numpy
            data = np.array(data, copy=True)

        arr = np.ascontiguousarray(data)
        # Prefer 'metadata'; fallback to 'original_metadata'
        meta = {}
        for k in ("metadata", "original_metadata"):
            v = d.get(k)
            if isinstance(v, dict):
                meta = v
                break
        return arr, meta

    def _extract_rsciio_array(obj):
        """
        Extract a NumPy array from rsciio reader output.
        Handles dicts with 'data', lists/tuples of such dicts, or arrays directly.
        Returns (np.ndarray, debug_info_str) or raises ValueError.
        """

        def _from_dict(d):
            if not isinstance(d, dict):
                return None
            if "data" in d:
                arr = d["data"]
                if isinstance(arr, np.ndarray):
                    return arr
            # common alternates some readers use
            for key in ("signals", "Signal", "items", "datasets"):
                if key in d:
                    v = d[key]
                    # try first element or iterate
                    if isinstance(v, (list, tuple)) and v:
                        for item in v:
                            got = _from_dict(item)
                            if got is not None:
                                return got
            return None

        # direct ndarray
        if isinstance(obj, np.ndarray):
            return obj, f"type=ndarray shape={obj.shape} dtype={obj.dtype}"

        # dict top-level
        if isinstance(obj, dict):
            arr = _from_dict(obj)
            if arr is not None:
                return arr, f"type=dict→ndarray shape={arr.shape} dtype={arr.dtype} keys={list(obj.keys())}"
            raise ValueError(f"Dict had no usable 'data' (keys={list(obj.keys())})")

        # list/tuple top-level
        if isinstance(obj, (list, tuple)):
            dbg = []
            for i, item in enumerate(obj):
                if isinstance(item, np.ndarray):
                    return item, f"type=list[{i}] ndarray shape={item.shape} dtype={item.dtype}"
                if isinstance(item, dict):
                    arr = _from_dict(item)
                    if arr is not None:
                        return arr, f"type=list[{i}] dict→ndarray shape={arr.shape} dtype={arr.dtype}"
                dbg.append(type(item).__name__)
            raise ValueError(f"List/tuple contained no ndarray/dict-with-data (inner types={dbg})")

        raise ValueError(f"Unsupported rsciio return type: {type(obj).__name__}")

    def load_blo():
        nonlocal data_array, circle_center, df_centers
        path = filedialog.askopenfilename(parent=root, filetypes=[("Blockfile", "*.blo"), ("All files", "*.*")])
        if not path:
            return

        def start_fn(cancel_flag, q):
            q.put(('status', "Reading .blo…"))
            if cancel_flag['stop']:
                return
            try:
                arr, meta = _read_blo_rsciio_interactive(path, parent=root, lazy=False, endianess="<")
                if arr is None:
                    q.put(('error', "Cancelled by user."))
                    return
            except Exception as e:
                q.put(('error', f"{type(e).__name__}: {e}"))
                return
            if cancel_flag['stop']:
                return
            q.put(('done', (arr, meta)))

        def _pump_done(payload):
            nonlocal data_array, circle_center, df_centers
            arr, meta = payload
            arr4d = _coerce_to_4d(arr, root, ask_scan_width)
            if arr4d is None:
                return
            data_array = arr4d
            H, W = data_array.shape[2], data_array.shape[3]
            circle_center[:] = [W // 2, H // 2]
            df_centers[:] = [list(circle_center)]  # reset DF centers
            current_scan_xy[:] = [data_array.shape[1] // 2, data_array.shape[0] // 2]
            update_pointer_image(*current_scan_xy)
            _metadata_set(meta)
            messagebox.showinfo("Load .blo", f"Loaded → navigator {data_array.shape[:2]}", parent=root)
            update_main_image(trigger_user_function=True)

        _load_with_progress(lambda cancel_flag, q: start_fn(cancel_flag, q), "Load .blo")

    def _extract_array_and_meta(obj):
        """
        Try to extract (ndarray, metadata_dict) from rsciio object structures.
        Looks into dicts/lists for 'data' and ('metadata' or 'original_metadata').
        """

        def _from_dict(d):
            if not isinstance(d, dict): return None, None
            arr = d.get("data")
            meta = d.get("metadata") or d.get("original_metadata")
            if isinstance(arr, np.ndarray):
                return arr, meta if isinstance(meta, dict) else {}
            # some formats nest deeper
            for key in ("signals", "Signal", "items", "datasets"):
                if key in d and isinstance(d[key], (list, tuple)):
                    for item in d[key]:
                        a, m = _from_dict(item)
                        if a is not None:
                            return a, m
            return None, None

        if isinstance(obj, np.ndarray):
            return obj, {}
        if isinstance(obj, dict):
            a, m = _from_dict(obj)
            if a is not None:
                return a, m
        if isinstance(obj, (list, tuple)):
            for item in obj:
                a, m = _extract_array_and_meta(item)
                if a is not None:
                    return a, m
        return None, {}

    def load_hspy():
        nonlocal data_array, circle_center, df_centers
        path = filedialog.askopenfilename(parent=root, filetypes=[("HyperSpy", "*.hspy"), ("All files", "*.*")])
        if not path:
            return

        def start_fn(cancel_flag, q):
            q.put(('status', "Reading .hspy…"))
            if cancel_flag['stop']:
                return
            try:
                obj = rs_hspy.file_reader(path)
                if cancel_flag['stop']:
                    return
                arr, meta = _extract_array_and_meta(obj)
                if arr is None:
                    raise ValueError("Could not find ndarray in .hspy file.")
                q.put(('done', (arr, meta)))
            except Exception as e:
                q.put(('error', f"{type(e).__name__}: {e}"))

        def _pump_done(payload):
            nonlocal data_array, circle_center, df_centers
            arr, meta = payload
            arr4d = _coerce_to_4d(arr, root, ask_scan_width)
            if arr4d is None:
                return
            data_array = arr4d
            H, W = data_array.shape[2], data_array.shape[3]
            circle_center[:] = [W // 2, H // 2]
            df_centers[:] = [list(circle_center)]  # reset DF centers
            current_scan_xy[:] = [data_array.shape[1] // 2, data_array.shape[0] // 2]
            update_pointer_image(*current_scan_xy)
            _metadata_set(meta)
            messagebox.showinfo("Load .hspy", f"Loaded → navigator {data_array.shape[:2]}", parent=root)
            update_main_image(trigger_user_function=True)

        _load_with_progress(start_fn, "Load .hspy")

    def load_data():
        """Loads a .npy file as a 4D data array."""
        nonlocal data_array, circle_center, df_centers
        file_path = filedialog.askopenfilename(parent=root, filetypes=[("NumPy Array", "*.npy")])
        if not file_path:
            return

        arr = load_npy_with_progress(file_path)
        if arr.ndim == 4:
            data_array = arr
        elif arr.ndim == 3:
            N, H, W = arr.shape
            guessed_scan_width = int(np.sqrt(N))
            scan_width = ask_scan_width(root, guessed_scan_width, N)
            if not scan_width:
                return
            if N % scan_width != 0:
                messagebox.showwarning(
                    "Load NumPy Array",
                    f"{N} frames not divisible by {scan_width}; last row will be padded.",
                    parent=root
                )
                missing = scan_width - (N % scan_width)
                pad = np.zeros((missing, H, W), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
                N = arr.shape[0]

            scan_height = N // scan_width
            data_array = arr.reshape(scan_height, scan_width, H, W)
        else:
            messagebox.showerror("Load NumPy Array",
                                 f"Expected 3D or 4D array, got shape {arr.shape}",
                                 parent=root)
            return

        H, W = data_array.shape[2], data_array.shape[3]
        circle_center[:] = [W // 2, H // 2]
        current_scan_xy[:] = [data_array.shape[1] // 2, data_array.shape[0] // 2]
        update_pointer_image(*current_scan_xy)
        df_centers[:] = [list(circle_center)]  # reset DF centers

        #messagebox.showinfo("Import Complete",
        #                    f"Loaded array with shape {data_array.shape[:2]} navigator.",
        #                    parent=root)
        update_function()

    def update_function(*args):
        """Regenerate and display the main image based on the selected function."""
        nonlocal main_image_pil, df_centers
        if data_array is None:
            return

        sel = selected_function.get()
        if sel not in functions:
            return

        print(f"update_function() called, sel = {sel}")

        # ---- Inline "Generating…" bar ----
        spinner_frame = tk.Frame(root)
        spinner_frame.grid(row=99, column=0, columnspan=2, pady=6, sticky="ew")
        lbl = tk.Label(spinner_frame, text="Generating…")
        lbl.grid(row=0, column=0, padx=8)
        bar = ttk.Progressbar(spinner_frame, mode="indeterminate", length=220)
        bar.grid(row=0, column=1, padx=8, sticky="ew")
        bar.start(15)

        def worker():
            try:
                print("Running analysis function...")
                if sel == "DF":
                    arr = VDF_multi(data_array, radius_value.get(), df_centers)
                elif sel == "VADF":
                    outer_radius = max(radius_value.get() + 1, outer_radius_value.get())
                    arr = functions[sel]["fn"](data_array,
                                               radius_value.get(),
                                               outer_radius,
                                               tuple(circle_center))
                else:
                    arr = functions[sel]["fn"](data_array,
                                               radius_value.get(),
                                               tuple(circle_center))

                #print(f"Analysis output shape: {arr.shape}")
                normalized = normalize_to_8bit(arr)
                result = Image.fromarray(normalized).convert("RGB")
            except Exception as e:
                result = e
            root.after(0, lambda: finish(result))

        def finish(result):
            nonlocal main_image_pil  # <<< CRUCIAL
            bar.stop()
            spinner_frame.destroy()
            if isinstance(result, Exception):
                messagebox.showerror("Analysis Error", str(result), parent=root)
                return
            main_image_pil = result
            update_pointer_image(current_scan_xy[0], current_scan_xy[1])
            update_main_image()

        threading.Thread(target=worker, daemon=True).start()

    def export_blo():
        """Prompt for pixel size + path, then write current data_array to .blo."""
        if data_array is None:
            messagebox.showwarning("Export .blo", "No data loaded.", parent=root);
            return

        px = ask_pixel_size_nm(root, guessed_nm=None)
        if px is None:  # cancelled
            return

        path = filedialog.asksaveasfilename(
            parent=root,
            defaultextension=".blo",
            filetypes=[("Blockfile", "*.blo"), ("All files", "*.*")],
            initialfile="exported.blo",
            title="Save .blo"
        )
        if not path:
            return

        try:
            # You can set bin_data_to_128=False if you want native DP size preserved.
            array2blo(data_array, px_nm=px, save_path=path,
                      intensity_scaling="crop", bin_data_to_128=True)
        except Exception as e:
            messagebox.showerror("Export .blo", f"Failed to write file:\n{e}", parent=root);
            return

        messagebox.showinfo("Export .blo", f"Saved:\n{path}", parent=root)

    def export_tiff_folder():
        """Threaded, cancellable export of each DP as 00001.tiff, 00002.tiff, ..."""
        if data_array is None:
            messagebox.showwarning("Export array to .tiffs", "No data loaded.", parent=root)
            return

        out_dir = filedialog.askdirectory(parent=root, title="Select output folder")
        if not out_dir:
            return

        Y, X, H, W = data_array.shape
        total = Y * X
        pad = len(str(total))  # zero-padding width

        # Progress window (modal)
        win = tk.Toplevel(root)
        win.title("Export TIFFs")
        win.transient(root)
        win.grab_set()
        tk.Label(win, text=f"Exporting {total} TIFFs…").pack(padx=10, pady=(10, 0))
        bar = ttk.Progressbar(win, orient="horizontal", length=360, mode="determinate", maximum=total)
        bar.pack(padx=10, pady=10)
        lbl = tk.Label(win, text=f"0 / {total} (0.0%)")
        lbl.pack(padx=10, pady=(0, 10))
        cancel_flag = {'stop': False}

        def _cancel():
            cancel_flag['stop'] = True
            lbl.config(text="Cancelling…")

        tk.Button(win, text="Cancel", command=_cancel).pack(pady=(0, 10))

        q = queue.Queue()

        def worker():
            try:
                idx = 0
                # Ensures directory exists
                os.makedirs(out_dir, exist_ok=True)
                for y in range(Y):
                    if cancel_flag['stop']:
                        q.put(('cancelled', None));
                        return
                    for x in range(X):
                        if cancel_flag['stop']:
                            q.put(('cancelled', None));
                            return
                        idx += 1
                        fname = os.path.join(out_dir, f"{idx:0{pad}d}.tiff")
                        # Write one DP (original dtype preserved)
                        tiff.imwrite(fname, data_array[y, x], photometric='minisblack')
                        # Post progress
                        if idx % 1 == 0:
                            q.put(('progress', idx))
                q.put(('done', total))
            except Exception as e:
                q.put(('error', f"{type(e).__name__}: {e}"))

        def pump():
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == 'progress':
                        i = payload
                        bar['value'] = i
                        lbl.config(text=f"{i} / {total} ({(i / total) * 100:.1f}%)")
                    elif kind == 'done':
                        win.grab_release();
                        win.destroy()
                        messagebox.showinfo("Export TIFFs", f"Exported {total} TIFFs to:\n{out_dir}", parent=root)
                        return
                    elif kind == 'cancelled':
                        win.grab_release();
                        win.destroy()
                        messagebox.showinfo("Export TIFFs", "Export cancelled.", parent=root)
                        return
                    elif kind == 'error':
                        win.grab_release();
                        win.destroy()
                        messagebox.showerror("Export TIFFs", payload, parent=root)
                        return
            except queue.Empty:
                pass
            root.after(33, pump)  # ~30 Hz UI updates

        threading.Thread(target=worker, daemon=True).start()
        pump()

    def on_function_change(event=None):
        """
        Show/hide controls that are specific to the selected function,
        then recompute the navigator and refresh overlays.
        """
        fn = selected_function.get()

        # Only show outer radius controls when VADF is selected
        if fn == "VADF":
            outer_radius_label.pack(side=tk.LEFT, padx=5, pady=5)
            outer_radius_entry.pack(side=tk.LEFT, padx=5, pady=5)
        else:
            outer_radius_label.pack_forget()
            outer_radius_entry.pack_forget()

        # Clamp outer >= inner when VADF is active
        if fn == "VADF":
            try:
                rin = int(radius_value.get())
                rout = int(outer_radius_value.get())
                if rout < rin:
                    outer_radius_value.set(rin)
            except Exception:
                pass

        update_function()

    # --- UI Layout ---
    SQUARE_CANVAS_SIZE = 512
    MIN_CANVAS_SIZE = 256

    # Top Frame for Controls
    top_frame = tk.Frame(root)
    top_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
    root.grid_rowconfigure(0, weight=0)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)



    # File I/O buttons
    tk.Button(top_frame, text="Load Numpy Array", command=load_data).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(top_frame, text="Load .tiff series", command=load_series).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(top_frame, text="Load .blo", command=load_blo).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(top_frame, text="Load .hspy", command=load_hspy).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(top_frame, text="Load metadata (.json)", command=load_metadata_json).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(top_frame, text="Export .blo", command=export_blo).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(top_frame, text="Export .tiff series", command=export_tiff_folder).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(top_frame, text="Save Images", command=save_images).pack(side=tk.LEFT, padx=5, pady=5)

    tk.Button(top_frame, text="Remove Last Marker", command=remove_last_marker).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(top_frame, text="Remove All Markers", command=clear_all_markers).pack(side=tk.LEFT, padx=5, pady=5)


    # Detector radius
    radius_label = tk.Label(top_frame, text="Detector Radius (px):")
    radius_label.pack(side=tk.LEFT, padx=5, pady=5)

    radius_entry = tk.Entry(top_frame, textvariable=radius_value, width=5)
    radius_entry.pack(side=tk.LEFT, padx=5, pady=5)
    radius_entry.bind("<Return>", lambda e: (clamp_radius(), "break"))
    radius_entry.bind("<FocusOut>", clamp_radius)

    # Function dropdown
    function_dropdown = ttk.Combobox(
        top_frame,
        textvariable=selected_function,
        values=list(functions.keys()),
        state="readonly"
    )
    function_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
    function_dropdown.bind("<<ComboboxSelected>>", on_function_change)

    # Outer radius (VADF only, shown dynamically)
    outer_radius_value = tk.IntVar(value=50)
    outer_radius_label = tk.Label(top_frame, text="Outer Radius (px):")
    outer_radius_entry = tk.Entry(top_frame, textvariable=outer_radius_value, width=5)

    # --- Main Frame with 2 panels ---
    main_frame = tk.Frame(root)
    main_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
    root.grid_rowconfigure(1, weight=1)

    # Left panel (navigator)
    left_panel = tk.Frame(main_frame)
    left_panel.grid(row=0, column=0, sticky="nsew")
    main_frame.grid_columnconfigure(0, weight=3)  # left takes more width
    main_frame.grid_rowconfigure(0, weight=1)

    main_canvas = tk.Canvas(left_panel, bg="white")
    main_canvas.pack(fill="both", expand=True)
    main_canvas.bind("<Button-1>", on_click)
    main_canvas.bind("<Motion>", _on_mouse_motion)
    main_canvas.bind("<Button-2>", toggle_mouse_motion)

    left_status_var = tk.StringVar(value="Mouse: (—, —)  |  motion: on")
    main_label = tk.Label(left_panel, textvariable=left_status_var)
    main_label.pack(fill="x")

    # Right panel (diffraction + metadata)
    right_panel = tk.Frame(main_frame)
    right_panel.grid(row=0, column=1, sticky="nsew")
    main_frame.grid_columnconfigure(1, weight=2)  # right panel narrower

    # Diffraction canvas stays square
    pointer_canvas = tk.Canvas(right_panel, bg="white",
                               width=SQUARE_CANVAS_SIZE, height=SQUARE_CANVAS_SIZE)
    pointer_canvas.pack(padx=5, pady=5)

    # --- Diffraction display controls (gamma/brightness/contrast/log) ---
    dp_ctrl = tk.Frame(right_panel)
    dp_ctrl.pack(fill="x", padx=8, pady=(0, 6))

    def _on_dp_adjust(_=None):
        # refresh diffraction view only
        if pointer_image_pil is not None:
            update_pointer_image(current_scan_xy[0], current_scan_xy[1])

    tk.Label(dp_ctrl, text="Gamma").grid(row=0, column=0, sticky="w")
    tk.Scale(dp_ctrl, variable=dp_gamma, from_=0.2, to=5.0, resolution=0.05,
             orient="horizontal", command=_on_dp_adjust).grid(row=0, column=1, sticky="ew")

    tk.Label(dp_ctrl, text="Brightness").grid(row=1, column=0, sticky="w")
    tk.Scale(dp_ctrl, variable=dp_brightness, from_=-100, to=100, resolution=1,
             orient="horizontal", command=_on_dp_adjust).grid(row=1, column=1, sticky="ew")

    tk.Label(dp_ctrl, text="Contrast").grid(row=2, column=0, sticky="w")
    tk.Scale(dp_ctrl, variable=dp_contrast, from_=0.2, to=3.0, resolution=0.02,
             orient="horizontal", command=_on_dp_adjust).grid(row=2, column=1, sticky="ew")

    tk.Checkbutton(dp_ctrl, text="Log intensity", variable=dp_log_intensity,
                   command=_on_dp_adjust).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

    dp_ctrl.grid_columnconfigure(1, weight=1)

    def _reset_dp_adjust():
        dp_gamma.set(1.0)
        dp_brightness.set(0.0)
        dp_contrast.set(1.0)
        dp_log_intensity.set(False)
        _on_dp_adjust()

    tk.Button(dp_ctrl, text="Reset", command=_reset_dp_adjust).grid(row=0, column=2, rowspan=4, padx=(8, 0), sticky="ns")


    def _keep_square(event):
        side = max(MIN_CANVAS_SIZE, min(event.width, event.height))
        if pointer_canvas.winfo_width() != side or pointer_canvas.winfo_height() != side:
            pointer_canvas.config(width=side, height=side)
        _refresh_pointer()  # redraw after resize

    pointer_canvas.bind("<Configure>", _keep_square)
    pointer_canvas.bind("<Button-1>", on_pointer_click)
    pointer_canvas.bind("<Button-3>", on_pointer_click)

    right_status_var = tk.StringVar(value="Scan: (—, —)  |  center: (—, —)")
    pointer_label = tk.Label(right_panel, textvariable=right_status_var)
    pointer_label.pack(fill="x")

    # Metadata Treeview fills remaining space
    meta_frame = tk.Frame(right_panel)
    meta_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

    cols = ("Value",)
    metadata_tree = ttk.Treeview(meta_frame, columns=cols, show="tree headings")
    metadata_tree.heading("#0", text="Key")
    metadata_tree.heading("Value", text="Value")

    xs = ttk.Scrollbar(meta_frame, orient="horizontal", command=metadata_tree.xview)
    ys = ttk.Scrollbar(meta_frame, orient="vertical", command=metadata_tree.yview)
    metadata_tree.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)

    metadata_tree.pack(side="left", fill="both", expand=True)
    ys.pack(side="right", fill="y")
    xs.pack(side="bottom", fill="x")

    # --- Resize-aware redraw bindings ---
    def _refresh_main(event=None):
        if main_image_pil is not None:
            update_main_image()

    def _refresh_pointer(event=None):
        if pointer_image_pil is not None:
            update_pointer_image(current_scan_xy[0], current_scan_xy[1])

    main_canvas.bind("<Configure>", _refresh_main)
    pointer_canvas.bind("<Configure>", _refresh_pointer)
    # Start the Tkinter main loop
    root.mainloop()

#visualiser()
if __name__ == "__main__":
    visualiser()