# Cotton Plant Mapper v38

Interactive map symbol scaling improved.

Boll, White Flower, Square and Cracked Boll are now normalised to a similar
visual size. The renderer crops excess transparent/light padding from each
source image before applying a shared display scale, so one symbol should no
longer appear much larger or smaller simply because its PNG canvas differs.

PDF export is unchanged.
