# language_annotations_notes.md

Scene notes from GPT-6 Astra (`us.openai.gpt-6-astra`, Bedrock Converse) given, per task, 6 head-camera + 3 right-wrist frames from two
demonstrations of `yam_mt9x200_v1` (one early, one late recording). These notes justify the nouns used in `randomized:` and flag where the
recorded prompt does not match what the demonstrations show. Generated 2026-11-05.

## HangMugs
- episodes: HangMugs_7, HangMugs_8
- recorded prompt accurate: False
  - The mug tree and three mugs are already arranged on the table when both demonstrations begin; placing the tree and spreading the mugs around it are not shown.
  - The hanging goal matches the visible manipulation, but demonstration 1 does not clearly show the cream-white mug being released onto the tree.
- what Astra saw: Both demonstrations start with a wooden mug tree already near the middle of a wooden table and three mugs around it: one dark brown, one red, and one cream-white. In demonstration 1, the right arm lifts the red mug and hangs it by its handle on a branch. The left arm hangs the dark brown mug on another branch, then lifts the cream-white mug and brings its handle toward an available branch; its final release is not clearly shown. In demonstration 2, the left arm hangs the dark brown mug first and the red mug second. The right arm then lifts the cream-white mug and hangs it on a right-side branch. All three mugs are hanging from the tree at the end of demonstration 2.
- objects: wooden mug tree with a round base and several branches; dark brown mug; red mug; cream-white mug; wooden table; left robot arm with a gripper; right robot arm with a gripper

## PlaceTeaBagInTeapot
- episodes: PlaceTeaBagInTeapot_10, PlaceTeaBagInTeapot_11
- recorded prompt accurate: True
  - The prompt correctly describes placing the tea bag into the teapot, but omits the left arm picking up the separate lid and covering the teapot afterward.
- what Astra saw: Both demonstrations start with a light-colored tea bag on a blue plate to the right of an open black teapot on a wooden table. The teapot has a tan-wrapped handle and a silver-colored opening rim; its round black lid rests separately to the left. The right gripper picks up the tea bag from the plate, carries it over the teapot, and releases it inside. The left gripper picks up the black lid and places it onto the teapot. In demonstration 1, the left arm retrieves the lid while the right arm inserts the bag; in demonstration 2, it retrieves the lid afterward. Both demonstrations end with the tea bag inside the covered teapot and the blue plate empty.
- objects: white to beige tea bag; blue plate; black teapot with a tan-wrapped handle and silver-colored opening rim; round black teapot lid; wooden table; left and right robot grippers

## SweepBeans
- episodes: SweepBeans_14, SweepBeans_15
- recorded prompt accurate: True
  - The tools are returned to the tray on the table rather than directly to the wooden tabletop.
  - Demonstration 1 does not visibly complete the dustpan return: it remains raised in the final frame.
- what Astra saw: Small tan beans are scattered near the center of a large dark gray rectangular tray on a wooden table. A gray dustpan with a lime-green lip and a gray hand broom with black bristles and lime-green accents lie beside them. In demonstration 1, the left arm holds and positions the dustpan while the right arm sweeps the beans into it. The left arm then lifts and tilts the loaded dustpan, and the broom is lowered onto the tray; the dustpan is still raised in the final frame. In demonstration 2, the right arm holds the dustpan while the left arm brushes the beans into it. Both tools are returned to the tray, with the collected beans remaining in the dustpan.
- objects: small tan beans; gray dustpan with a lime-green lip; gray hand broom with black bristles and lime-green accents; large dark gray rectangular tray; light wooden table

## MakeCoffee
- episodes: MakeCoffee_8, MakeCoffee_9
- recorded prompt accurate: True
  - The word 'cup' refers to the small coffee pod, not the drinking mug; 'coffee pod' would be clearer.
  - Opening the coffee maker's lid to access the pod holder is visible but not explicitly included in the prompt.
  - The prompt accurately describes the demonstrated end state; brewing or dispensing coffee is not shown.
- what Astra saw: Both demonstrations start with a black and silver single-serve coffee maker, a small white coffee pod with a printed foil lid, and a cream-colored handled mug on a wooden worktop. The left arm stays near the coffee maker's upper left side while the right arm opens or accesses the top compartment, picks up the pod, inserts it into the circular holder, and closes the lid. The right gripper then grasps the mug by its handle and moves it onto the black drip tray beneath the spout. In the second demonstration, the mug tips sideways during handling before being positioned upright. Both demonstrations end with the lid closed and the mug beneath the spout; no coffee dispensing is visible.
- objects: black and silver single-serve coffee maker with a hinged top lid; black drip tray beneath the coffee spout; white coffee pod with a multicolored printed foil lid; cream-colored ceramic mug with a handle; light wooden worktop; black robot grippers on white robot arms

## ToastBread
- episodes: ToastBread_16, ToastBread_17
- recorded prompt accurate: True
  - The prompt omits the visible manipulation of the toaster's front lever.
  - The frames show loading and removing the bread, but do not confirm that it has finished toasting.
- what Astra saw: Both demonstrations start with a slice of white bread with golden-brown crust on a light cutting board with black edges, beside a white two-slot toaster and an empty blue-gray plate on a wooden table. In demonstration 1, the board is to the toaster's right and the plate is to its left. The right arm picks up the bread, inserts it upright into a toaster slot, and reaches toward the front lever. The left arm then removes the bread and lays it flat on the plate. In demonstration 2, the board is on the left and the plate is on the right. The left arm loads the bread and manipulates the toaster's front lever; the right arm removes the bread and places it on the plate. Both end with the bread on the plate and the cutting board and toaster slots empty. A completed heating cycle or change in browning cannot be confirmed from these frames.
- objects: slice of white bread with golden-brown crust; white two-slot toaster with black slot openings and a front lever; light cutting board with black edges; round blue-gray plate; wooden table

## CleanUpSpill
- episodes: CleanUpSpill_5, CleanUpSpill_6
- recorded prompt accurate: True
- what Astra saw: Both demonstrations start with a blue mug tipped on its side beside a brown liquid spill near the center of a large gray tray on a wooden table. A white towel lies to the left of the tray. The right arm picks up the mug and sets it upright near the tray's upper-right corner. The left arm picks up the towel, presses it onto the spill, and wipes across the tray to absorb the liquid. It then moves the stained towel back to the left side. The mug remains upright off to the side, and the pooled liquid is removed; faint wipe marks remain visible in the first demonstration.
- objects: blue mug with a handle and silver-colored rim; brown liquid spill; white towel; large gray tray with a raised edge; wooden table

## BuildSprinkler
- episodes: BuildSprinkler_3, BuildSprinkler_4
- recorded prompt accurate: True
  - The prompt correctly describes the overall task but leaves the insert-fitting and lock-ring-fitting steps implicit.
  - Returning the sprinkler to the table is visible in the second demonstration but not in the supplied ending frames of the first.
- what Astra saw: On a light wooden table are sprinkler parts: a black cylindrical body with a pale end, a narrower black ribbed insert with white or silver-colored ends, and a thick black lock ring. The left gripper holds and steadies the sprinkler body while the right gripper brings the insert into its opening. The right gripper then picks up the lock ring, aligns it with the exposed end, and fits it onto the assembly while the left gripper supports the body. Both arms reposition the sprinkler during fitting. In the second demonstration, the right arm lowers the assembled sprinkler back onto the tabletop and the left gripper withdraws. The first demonstration ends with the assembly still held between the grippers, so its placement on the table is not shown.
- objects: black cylindrical sprinkler body with a pale end; black ribbed sprinkler insert with white or silver-colored ends; thick black lock ring; light wooden table

## SortUtensils
- episodes: SortUtensils_12, SortUtensils_13
- recorded prompt accurate: True
  - The first recorded prompt is cut off after 'knives'; the other prompts accurately describe the visible sorting task.
  - No additional task steps are visibly required beyond collecting the loose utensils and grouping them by type in the tray.
- what Astra saw: Both demonstrations show the same sequence. Silver knives, forks, and spoons with dark blue handles are scattered around a dark gray divided tray on a light wooden table; some utensils are already in the tray. Both robot arms pick up loose utensils and lower them into the tray, grouping matching types together. The left arm visibly places a knife and a spoon, while the right arm handles spoons and forks and places the final fork. By the end, the surrounding tabletop is clear of loose utensils and the utensils are grouped in the tray's compartments.
- objects: silver forks with dark blue handles; silver knives with dark blue handles; silver spoons with dark blue handles; dark gray divided tray; light wooden table; two robot arms with black grippers

## BikeRotorToolUse
- episodes: BikeRotorToolUse_1, BikeRotorToolUse_2
- recorded prompt accurate: True
- what Astra saw: A bicycle wheel with a black knobby tire, black spokes and a silver disc brake rotor lies flat on a light wooden workbench, rotor side up. The robot brings a long silver-shafted tool with a dark socket end to the lock ring at the wheel hub. The right gripper holds the tool engaged with the lock ring while the left gripper steadies and turns the wheel to tighten it. The arms reposition during tightening, then lift the tool away, leaving the rotor and lock ring on the wheel hub.
- objects: bicycle wheel with a black knobby tire, black rim and black spokes; silver disc brake rotor; black wheel hub; dark metal lock ring at the rotor center; long silver-shafted tool with a dark socket end; light wooden workbench
