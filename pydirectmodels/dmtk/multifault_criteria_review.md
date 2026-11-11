# Multi-fault versus multi-segment ruptures: a comparison of three approaches

This note compares how Al Atik et al. (2023), Weatherill & Lilienkamp (2023) and our implementation (`dmtk.py`) decide whether an earthquake rupture from a fault-system rupture forecast is treated as one multi-segment rupture or as a multi-fault rupture made of separate fault strands, and how each then prepares the rupture for directivity modelling. It ends with a critical review of each approach and of the assumptions they share.

Sources: Al Atik, Gregor, Mazzoni & Bozorgnia (2023), *Directivity-based PSHA for the State of California, Report 2: Directivity Implementation*, GIRS-2023-05, Section 5; Weatherill & Lilienkamp (2023), *Capturing Directivity in PSHA for New Zealand*, BSSA 114(1), 373–398; and `dmtk.py` as developed in this project.

## Comparison

| Aspect | Al Atik et al. (2023) | Weatherill & Lilienkamp (2023) | Ours (`dmtk.py`) |
|---|---|---|---|
| Rupture set | UCERF3 (1,634,466 ruptures) | NZ NSHM 2022 crustal ERF (3,884 unique ruptures over 324 branches) | NZ NSHM 2022 rupture set |
| Core distinction | Multi-fault: each strand has its own hypocentre; multi-segment: one hypocentre for the whole rupture | Same | Same |
| Classification rule | Δdip > 50° **and** Δrake₁Q > 30° between consecutive subsections | Δdip > 50° **and** Δrake₁Q > 30° **and** gap > 10 km between consecutive (micro)segments | Step 1: Weatherill & Lilienkamp rule between consecutive subsections; step 2: a parallel/branching strand. Multi-fault if either applies (Al Atik rule available as an option) |
| Role of the 10 km gap | Places extra breaks, only within ruptures already multi-fault | Required for classification | Required for classification and breaks (as Weatherill & Lilienkamp) |
| Dip change | Plain absolute difference | Not specified beyond "change in dip angle" | Plain absolute difference by default; dip-direction-aware option |
| Parallel / overlapping strands | Not considered | Not considered | Along-strike overlap > 10 km → parallel (Δstrike ≤ 30°) or branching; makes the rupture multi-fault |
| Long jumps between strands | Not discussed | Not discussed | Jumps > 15 km end a chain; strands lying beyond an end of the main strand are re-attached, so a gap alone is not a break |
| Order of subsections | Rupture-set order | Consecutive microsegments as listed in the rupture | Rebuilt from geometry; independent of input order |
| Trace geometry | Spline smoothing of multi-segment traces with gaps (latitude as a function of longitude or vice versa) | GC2 via the Bayless et al. (2020) model; no smoothing described | Parametric spline per contiguous strand piece; never across parallel strands |
| Subfault magnitude | Total moment split by strand area | Total moment split by subfault area | Same, using fault-plane area (length × down-dip width) |
| Subfault dip, rake, Ztor | Area-weighted averages | Rake: area-weighted mean | Per section available; strand averages not yet computed |
| Subfault length, distances | Computed per strand | Per subfault, through GC2 | Strand traces resampled every 1 km |
| Hypocentres | Uniform grid per strand: 3 down-dip; 3, 5 or 10 along strike for strands ≤ 20 km, 20–50 km, > 50 km | 250 samples per rupture: along strike from Watson-Lamprey (2018), down-dip from Melgar & Hayes (2019), sampled independently per subfault | Not yet implemented |
| Combining subfaults at a site | All hypocentre combinations equally likely; full weight to the hypocentre with the largest directivity predictor | Maximum directivity parameter across subfaults | Not yet implemented |
| Directivity model | Several models within Haz45 | Bayless et al. (2020) only | Not yet chosen |
| Outcome | 178,407 of 1,634,466 ruptures multi-fault (≈ 11%), 2–6 strands each | 69 of 3,884 ruptures multi-fault (≈ 1.8%) | Not yet run on the full set. Example: rupture 332805 is multi-fault through a parallel strand (step 2), multi-segment under both published rules |

## Critical review

### Al Atik et al. (2023)

The method's central idea is sound and physically motivated: directivity builds up only while a rupture propagates continuously, so a rupture that jumps between faults should have a hypocentre on each strand. Its two classification conditions target the most obvious kind of discontinuity, a change in faulting style at a junction. Using the gap only to refine breaks, rather than to classify, is also deliberate: the report shows ruptures with gaps of nearly 30 km that remain multi-segment.

The weaknesses lie in what the rule cannot see. It responds only to changes in mechanism, so two parallel strike-slip strands with identical dip and rake pass as one continuous rupture, however far apart they are or however long they overlap. That case is then made worse by the geometry step: smoothing a trace that runs across two side-by-side strands produces a line between them that does not exist, distorting length, strike and the along-strike coordinate that the directivity predictor depends on. Smoothing latitude as a function of longitude (or the reverse) also requires a switch for traces that turn, which a parametric spline avoids.

The thresholds of 50°, 30° and 10 km are judgement values rather than ones derived from rupture physics or data, and the dip change ignores dip direction, so a fault dipping 40° east next to one dipping 40° west registers no change. Finally, giving full weight to the hypocentre with the largest directivity predictor in each combination is a conservative choice that pushes amplification upward for multi-fault ruptures, and sampling hypocentres independently on each strand ignores the fact that rupture on a later strand is triggered by the earlier one.

### Weatherill & Lilienkamp (2023)

The paper's main contribution is computational: precomputing mean and variance of amplification for every unique rupture and compressing them into an overfitted neural network makes directivity feasible for a national PSHA. Its geometry handling is also cleaner than smoothing, because GC2 defines continuous along- and across-strike coordinates directly on the segmented trace. The authors are candid about limitations, including that the independent sampling of hypocentres across subfaults is physically questionable and that current directivity models are not calibrated for ruptures this complex or for magnitudes above Mw 8.

The classification step, however, differs from the method it cites. Requiring a gap greater than 10 km in addition to both angle conditions is stricter than Al Atik's rule, and the paper attributes this combined rule to Al Atik et al. (2023). The consequence is visible in the outcome: only 69 of 3,884 ruptures (about 1.8%) are multi-fault. A rupture whose mechanism changes sharply at a junction with no gap, for example a strike-slip fault running directly into a thrust, is treated as multi-segment with a single hypocentre, and the Bayless et al. (2020) model is then applied to a surface of mixed mechanism, which it was not designed for. Parallel strands are not considered here either.

Several details are left unstated. "Consecutive segments" is used for the comparison, but it is not clear whether this means subsections or parent segments, and the order in which they are compared is taken from the rupture list. The area-weighted mean rake is not said to be circular, which matters for rakes near ±180°. Because all hazard results flow through the precomputed fields, any classification choice is fixed into the trained network and cannot be revisited without regenerating the fields and retraining.

### Our approach (`dmtk.py`)

Our implementation adopts the Weatherill & Lilienkamp rule as its first step, so that its mechanism-and-gap classification is directly comparable with the NZ NSHM 2022 work, and then applies a geometric test for strands that run alongside or branch from the main strand. Al Atik's rule remains available as an option for comparison. That addition addresses the clearest gap in both published methods, and rupture 332805 shows it matters in practice: the Ohariu–Pukerua strand overlaps the Wellington fault for about 22 km at a separation of about 4 km, yet both published rules treat the rupture as one continuous trace. The implementation also records why each rupture is multi-fault, rebuilds the along-rupture order from geometry so results do not depend on input order, uses circular means for rake and dip direction, computes strand magnitudes from fault-plane rather than surface-projected areas, and never fits a spline across parallel strands.

Its limitations are equally clear. The parallel-strand test is our own extension, not part of any published method, and its thresholds (10 km overlap, 30° strike difference, 15 km maximum jump, 2 km allowed backtrack) are judgement values chosen for resolution and consistency, not calibrated. The greedy strand chaining is not guaranteed to pick the main strand a geologist would, particularly at Y-shaped junctions, and it has been tested on only a handful of ruptures. Rebuilding order from geometry is robust to messy inputs but may disagree with the rupture set's own section order, which we do not yet check. The upper edge is taken as vertices 0–1 of each polygon, verified only on an 82-section sample; the supplied polygons also appear mirrored about the trace for many faults, which does not affect the upper edge but does affect anything using the footprint. Distances use a single local projection per rupture, gaps are measured between edge endpoints rather than densified edges, and strand dip, rake and Ztor are not yet averaged. Above all, the method stops before the directivity step: hypocentres, the subfault combination rule and the choice of directivity model are still to be implemented, and it has not been run on the full rupture set.

### Shared assumptions

All three approaches inherit the same deeper uncertainties. The thresholds that separate multi-fault from multi-segment ruptures are not derived from observations of how directivity behaves across fault junctions, because such observations are scarce. Hypocentres on different subfaults are treated as independent, although rupture dynamics suggests otherwise. The directivity models themselves were calibrated on simpler ruptures, mostly strike-slip and below Mw 8, so applying them to complex multi-fault ruptures is an extrapolation whichever classification is used. And in every case the subfault magnitude comes from splitting moment by area, which assumes uniform average slip across strands.

## Recommendations

1. Report how many NZ ruptures are multi-fault under the adopted rule (Weatherill & Lilienkamp, then the geometric test), and separately under each step and under Al Atik's rule. `classify_rupture` returns all of these flags for every rupture. Check that step 1 alone reproduces Weatherill & Lilienkamp's 69 multi-fault ruptures; a mismatch would point to differences in section order or gap measurement.
2. Run sensitivity cases on the parallel-strand thresholds (overlap 5, 10, 15–20 km; strike difference 20°, 30°, 45°) and on the maximum jump distance, and report the change in the number of multi-fault ruptures.
3. Compare the geometric section order with the rupture set's own order, and inspect by eye a sample of ruptures where they differ, along with the most complex ruptures (most strands, largest gaps, most segments).
4. Complete the strand properties (area-weighted dip, circular-mean rake, Ztor) before implementing hypocentres, so the strands passed to the directivity model are fully described.
5. When describing the method, state clearly that step 1 follows the rule as applied by Weatherill & Lilienkamp to the NZ NSHM, which is stricter than Al Atik's report, and that the geometric step is our extension.
