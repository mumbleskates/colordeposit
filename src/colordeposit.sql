CREATE EXTENSION IF NOT EXISTS cube;  /* for kd-tree indexing */

CREATE TABLESPACE ramdisk LOCATION '/tmp/tspace';


CREATE UNLOGGED TABLE IF NOT EXISTS frontier(
  x smallint,
  y smallint,
  PRIMARY KEY (x, y),
  val cube
) TABLESPACE ramdisk;
CREATE INDEX IF NOT EXISTS frontier_ix_value ON frontier USING GIST (val) TABLESPACE ramdisk;
