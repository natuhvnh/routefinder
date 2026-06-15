import matplotlib.pyplot as plt
import time
from py3dbp.packer import Packer
from py3dbp.item import Item
from py3dbp.bin import Bin
from py3dbp.visualizer import Visualizer


def get_ldm(packer):
    max_depth = 0
    for b in packer.bins:
        if len(b.items) > 0:
            for item in b.items:
                item_end_z = float(item.position[2]) + float(item.get_dimension()[2])
                if item_end_z > max_depth:
                    max_depth = item_end_z
    return max_depth


def visualize_bin(packer, title_prefix=""):
    for b in packer.bins:
        print(f"Bin: {b.name} | Total Items: {len(b.items)}")
        visualizer = Visualizer(b)

        fig = visualizer.plot_box_and_items(
            title=f"{title_prefix} {b.name} - {len(b.items)} Items",
            alpha=1.0,
            write_num=True,
            fontsize=8,
        )

        ax = plt.gca()

        ax.set_xlim(0, float(b.width))
        ax.set_ylim(0, float(b.height))
        ax.set_zlim(0, float(b.depth))

        ax.set_box_aspect((float(b.width), float(b.height), float(b.depth)))
        ax.view_init(elev=0, azim=150, roll=-90)
        ax.scatter(0, 0, 0, color="red", s=400, marker="*", depthshade=False, zorder=10)
        # ==========================================
        fig = plt.gcf()
        fig.set_size_inches(16, 8)

        ax.set_xlabel("Width", labelpad=15)
        ax.set_ylabel("Height", labelpad=15)
        ax.set_zlabel("Length", labelpad=15)

        plt.tight_layout()
        plt.show()


def optimize_ldm_bounded(
    bin_width,
    bin_height,
    max_depth,
    max_weight,
    items_data,
    tol=20.0,
    support_surface_ratio=0.25,
):
    """
    Returns (initial_packer, best_packer).
    Searches within [initial_LDM - max_item_length, initial_LDM].
    """
    # 1. INITIAL UNCONSTRAINED RUN
    initial_packer = Packer()
    initial_bin = Bin(
        "Full-Trailer", (bin_width, bin_height, float(max_depth)), max_weight, 0, 0
    )
    initial_packer.add_bin(initial_bin)

    for item_kwargs in items_data:
        initial_packer.add_item(Item(**item_kwargs))

    initial_packer.pack(
        bigger_first=True,
        fix_point=True,
        check_stable=True,
        support_surface_ratio=support_surface_ratio,
    )

    if len(initial_packer.unfit_items) > 0:
        return None, None

    initial_ldm = get_ldm(initial_packer)

    # 2. CALCULATE LOGICAL BOUNDS
    # Find the absolute longest dimension of any item (since it can be rotated)
    max_item_length = max([max(item["whd"]) for item in items_data])

    max_d = initial_ldm
    # Ensure min_d doesn't drop below 0 just in case
    min_d = max(0.0, initial_ldm - max_item_length)

    # print(f"-> Initial Lazy LDM: {initial_ldm:.2f} mm")
    # print(f"-> Max Item Length: {max_item_length:.2f} mm")
    # print(
    #     f"-> Search Window: [{min_d:.2f} mm to {max_d:.2f} mm] with {tol} mm tolerance"
    # )

    best_packer = initial_packer
    iterations = 0
    items = [Item(**item_kwargs) for item_kwargs in items_data]

    # 3. BOUNDED BINARY SEARCH
    while (max_d - min_d) >= tol:
        iterations += 1
        mid_d = (min_d + max_d) / 2.0

        test_packer = Packer()
        test_bin = Bin(
            "Shrinking-Run", (bin_width, bin_height, mid_d), max_weight, 0, 0
        )
        test_packer.add_bin(test_bin)

        for item in items:
            test_packer.add_item(item)

        test_packer.pack(
            bigger_first=True,
            fix_point=True,
            check_stable=True,
            support_surface_ratio=support_surface_ratio,
        )

        if len(test_packer.unfit_items) == 0:
            best_packer = test_packer
            max_d = mid_d  # Fits! Try tighter.
        else:
            min_d = mid_d  # Doesn't fit. Give it more room.

    # print(f"-> Search completed in {iterations} iterations.")
    return initial_packer, best_packer


def items_packing(
    loading_items, equipment_whd, equipment_weight, num_stack, unit_type, visualize=True
):
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    start = time.time()
    #
    if unit_type == "Volume":
        best_ldm, init_ldm = float("inf"), float("inf")
        rotation = [0, 1, 2, 3, 4, 5]
        items_config = []
        for index, i in enumerate(loading_items):
            for v in range(i["quantity"]):
                items_config.append(
                    {
                        # 'partno': f'P-{i['id']}-{str(v)}',
                        "partno": f"P-{v}",
                        "group": "UK-Standard",
                        "type": "cube",
                        "whd": (i["width"], i["height"], i["length"]),
                        "weight": int(i["weight"] / i["quantity"]),
                        "loadbear": 1000,
                        "stackable": True,
                        "priority": 1,
                        "rotations": rotation,
                        "upsidedown": True,
                        "color": COLORS[index % len(COLORS)],
                    }
                )
        # print("Starting bounded LDM optimization...")
        initial_packer, best_packer = optimize_ldm_bounded(
            bin_width=equipment_whd[0],
            bin_height=equipment_whd[1],
            max_depth=equipment_whd[2],
            max_weight=equipment_weight,
            items_data=items_config,
            tol=200.0,
        )
        if best_packer is None:
            # raise Exception(
            #     "Error: Could not pack items even at maximum trailer depth."
            # )
            print("Could not pack items even at maximum trailer depth")
        else:
            # Get LDMs for comparison
            init_ldm = get_ldm(initial_packer)
            best_ldm = get_ldm(best_packer)

    elif unit_type == "Pallet":
        rotation = [0, 3]
        if num_stack is not None and num_stack <= 3:
            max_item_height = equipment_whd[1] / num_stack
        else:
            max_item_height = 0
        best_ldm, init_ldm = float("inf"), float("inf")
        best_packer, initial_packer = None, None
        for swap_wd in [True, False]:
            items_config = []
            for index, i in enumerate(loading_items):
                for v in range(i["quantity"]):
                    items_config.append(
                        {
                            # 'partno': f'P-{i['id']}-{str(v)}',
                            "partno": f"P-{v}",
                            "group": "UK-Standard",
                            "type": "cube",
                            "whd": (
                                (
                                    i["width"],
                                    max(i["height"], max_item_height),
                                    i["length"],
                                )
                                if swap_wd
                                else (
                                    i["length"],
                                    max(i["height"], max_item_height),
                                    i["width"],
                                )
                            ),
                            "weight": int(i["weight"] / i["quantity"]),
                            "loadbear": 1000,
                            "stackable": (
                                True
                                if (
                                    i["stack_on_top"] == "Y"
                                    and i["stack_on_other"] == "Y"
                                )
                                else False
                            ),
                            "priority": 1,
                            "rotations": rotation,
                            "upsidedown": True,
                            "color": COLORS[index % len(COLORS)],
                        }
                    )
            # print("Starting bounded LDM optimization...")
            initial_packer_output, best_packer_output = optimize_ldm_bounded(
                bin_width=equipment_whd[0],
                bin_height=equipment_whd[1],
                max_depth=equipment_whd[2],
                max_weight=equipment_weight,
                items_data=items_config,
                tol=200.0,
            )
            if best_packer_output is not None:
                ldm = get_ldm(best_packer_output)
                if ldm < best_ldm:
                    best_ldm = ldm
                    init_ldm = get_ldm(initial_packer_output)
                    best_packer = best_packer_output
                    initial_packer = initial_packer_output
        if best_packer is None:
            # raise Exception(
            #     "Error: Could not pack items even at maximum trailer depth."
            # )
            print("Could not pack items even at maximum trailer depth")
    # print("\n--- RESULTS ---")
    # print(f"Initial LDM:   {init_ldm/1000:.2f} m ({init_ldm:.2f} mm)")
    # print(f"Optimized LDM: {best_ldm/1000:.2f} m ({best_ldm:.2f} mm)")
    # print(f"Space Saved:   {(init_ldm - best_ldm)/1000:.2f} m")
    # print(f"\nTotal Time: {time.time() - start:.4f}s")
    # Visualize Both
    if visualize:
        print("\nVisualizing Initial Run...")
        visualize_bin(initial_packer)
        print("\nVisualizing Optimized Run...")
        visualize_bin(best_packer, title_prefix="[OPTIMIZED]")
    return best_ldm, init_ldm