import { drizzle } from "drizzle-orm/node-postgres";
import { Client } from "pg";
import duckdb from "duckdb";
import * as parquet from "parquetjs-lite";
import path from "path";
import fs from "fs";
import { sourceTable } from "./schema";

const PG_CONFIG = {
  host: process.env.HOST || "localhost",
  database: process.env.DATABASE || "postgres",
  user: process.env.USERNAME || "postgres",
  password: process.env.PASSWORD || "postgres",
  port: parseInt(process.env.PORT! || "5432"),
};

const DATA_DIR = path.resolve(__dirname, "../data");
const DUCKDB_FILE = path.join(DATA_DIR, "session_data.duckdb");
const LAST_TIMESTAMP_FILE = path.join(DATA_DIR, "last_timestamp.txt");

// --- Helper functions to store/retrieve last timestamp ---
function getLastTimestamp(): Date {
  if (!fs.existsSync(LAST_TIMESTAMP_FILE)) return new Date(0);
  const tsStr = fs.readFileSync(LAST_TIMESTAMP_FILE, "utf-8");
  return new Date(tsStr);
}

function setLastTimestamp(ts: Date) {
  fs.writeFileSync(LAST_TIMESTAMP_FILE, ts.toISOString(), "utf-8");
}

// --- Main script ---
async function main() {
  // 1. Connect to Postgres via Drizzle
  const pgClient = new Client(PG_CONFIG);
  await pgClient.connect();
  const db = drizzle(pgClient);

  const lastTimestamp = getLastTimestamp();
  console.log("Last processed timestamp:", lastTimestamp.toISOString());

  // 2. Pull new rows
  const rows = await db
    .select()
    .from(sourceTable)
    .where(sourceTable.createdAt.gt(lastTimestamp));

  if (rows.length === 0) {
    console.log("No new rows to process.");
    await pgClient.end();
    return;
  }

  console.log(`Fetched ${rows.length} new rows.`);

  // 3. Write new rows to temporary Parquet
  const tempParquet = path.join(DATA_DIR, "temp.parquet");
  const schema = new parquet.ParquetSchema({
    id: { type: "INT64" },
    name: { type: "UTF8" },
    value: { type: "INT64" },
    created_at: { type: "TIMESTAMP_MILLIS" },
  });

  const writer = await parquet.ParquetWriter.openFile(schema, tempParquet);
  for (const row of rows) {
    await writer.appendRow({
      id: row.id,
      name: row.name,
      value: row.value,
      created_at: row.createdAt,
    });
  }
  await writer.close();

  // 4. Append to persistent DuckDB
  const duck = new duckdb.Database(DUCKDB_FILE);
  const conn = duck.connect();

  // Create table if it doesn't exist
  await new Promise<void>((resolve, reject) => {
    conn.run(
      `CREATE TABLE IF NOT EXISTS sessions_data (
        id BIGINT,
        name VARCHAR,
        value BIGINT,
        created_at TIMESTAMP
      )`,
      (err) => (err ? reject(err) : resolve()),
    );
  });

  // Append new rows from Parquet
  await new Promise<void>((resolve, reject) => {
    conn.run(
      `INSERT INTO sessions_data SELECT * FROM read_parquet('${tempParquet}')`,
      (err) => (err ? reject(err) : resolve()),
    );
  });

  console.log(`Appended ${rows.length} rows to DuckDB.`);

  // 5. Update last processed timestamp
  const maxTimestamp = new Date(
    Math.max(...rows.map((r) => r.createdAt.getTime())),
  );
  setLastTimestamp(maxTimestamp);
  console.log("Updated last timestamp to:", maxTimestamp.toISOString());

  // 6. Optional: verify total rows
  await new Promise<void>((resolve, reject) => {
    conn.all(
      "SELECT COUNT(*) AS total_rows FROM sessions_data",
      (err, result) => {
        if (err) reject(err);
        else {
          console.log("Total rows in DuckDB:", result[0].total_rows);
          resolve();
        }
      },
    );
  });

  // 7. Clean up temp file and close connections
  fs.unlinkSync(tempParquet);
  await pgClient.end();
}

main().catch(console.error);
