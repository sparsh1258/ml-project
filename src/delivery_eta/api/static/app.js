"use strict";

const $ = (id) => document.getElementById(id);
const form = $("order");
const SCALE_END = 60;

const LABELS = {
  distance_km: "Distance",
  order_hour: "Time of day",
  rider_age: "Rider's age",
  rider_rating: "Rider's rating",
  vehicle_condition: "Vehicle condition",
  other_deliveries: "Other orders carried",
  vehicle: "Vehicle",
  weather: "Weather",
  traffic: "Traffic",
  festival: "Festival day",
  city: "Area",
  order_type: "Order",
};
const NUMBERS = ["vehicle_condition", "other_deliveries", "rider_rating", "rider_age"];
const CHOICES = ["order_type", "traffic", "weather", "city", "vehicle"];

/** The order as the API wants it. Blank fields are left out: they are unknown. */
function readOrder() {
  const order = { distance_km: Number($("distance").value) };
  // The date doesn't matter to the model, only the hour. No offset means Indian time.
  order.ordered_at = `2000-01-01T${$("time").value}`;
  for (const name of CHOICES) if ($(name).value) order[name] = $(name).value;
  for (const name of NUMBERS) if ($(name).value !== "") order[name] = Number($(name).value);
  if ($("festival").value) order.festival = $("festival").value === "yes";
  return order;
}

function item(tag, text, className) {
  const el = document.createElement(tag);
  el.textContent = text;
  if (className) el.className = className;
  return el;
}

function showProblem(text) {
  $("problem").textContent = text;
  $("problem").hidden = false;
  $("answer").classList.add("stale");
}

function showEstimate(e) {
  $("problem").hidden = true;
  $("answer").classList.remove("stale");
  $("minutes").querySelector("strong").textContent = Math.round(e.minutes);
  const low = Math.round(e.low);
  const high = Math.round(e.high);
  $("range-text").textContent = `Most likely ${low} to ${high} min. About 8 in 10 orders land in their range.`;

  const percent = (minutes) => Math.min(100, (Math.max(0, minutes) / SCALE_END) * 100);
  $("scale-range").style.left = `${percent(e.low)}%`;
  $("scale-range").style.width = `${percent(e.high) - percent(e.low)}%`;
  $("scale-mark").style.left = `${percent(e.minutes)}%`;

  const notes = e.warnings.map((text) => item("li", text, "warning"));
  if (e.missing.length) {
    const names = e.missing.map((name) => LABELS[name].toLowerCase()).join(", ");
    notes.push(item("li", `Not known: ${names}. The range is wider because of it.`));
  }
  $("notes").replaceChildren(...notes);

  const shown = e.factors.filter((f) => Math.abs(f.minutes) >= 0.1).slice(0, 7);
  const largest = Math.max(...shown.map((f) => Math.abs(f.minutes)), 4);
  $("factors").replaceChildren(
    ...shown.map((f) => {
      const row = item("li", "", f.minutes > 0 ? "slower" : "faster");
      const unknown = e.missing.includes(f.input) ? " (not known)" : "";
      const bar = item("span", "", "bar");
      const fill = document.createElement("i");
      fill.style.width = `${(Math.abs(f.minutes) / largest) * 50}%`;
      bar.append(fill);
      const sign = f.minutes > 0 ? "+" : "−";
      row.append(
        item("span", LABELS[f.input] + unknown),
        bar,
        item("span", `${sign}${Math.abs(f.minutes).toFixed(1)} min`, "amount"),
      );
      return row;
    }),
  );
}

let latest = null;

async function update() {
  $("distance-value").textContent = `${$("distance").value} km`;
  if (!form.checkValidity()) {
    showProblem("One of the fields is outside its allowed range.");
    return;
  }
  latest?.abort();
  const request = (latest = new AbortController());
  const headers = { "Content-Type": "application/json" };
  if ($("key").value) headers["X-API-Key"] = $("key").value;
  try {
    const response = await fetch("/api/v1/predict?explain=true", {
      method: "POST",
      headers,
      body: JSON.stringify(readOrder()),
      signal: request.signal,
    });
    const body = await response.json();
    if (response.ok) showEstimate(body);
    else if (response.status === 401) {
      $("key-box").hidden = false;
      showProblem("Enter the API key for this server to get estimates.");
    } else if (body.problems) {
      showProblem(body.problems.map((p) => `${LABELS[p.where] ?? p.where}: ${p.message}`).join(" "));
    } else showProblem(body.detail ?? "The server couldn't answer.");
  } catch (error) {
    if (error.name !== "AbortError") showProblem("Can't reach the server. Check your connection and try again.");
  }
}

let timer = 0;
form.addEventListener("input", () => {
  $("distance-value").textContent = `${$("distance").value} km`;
  clearTimeout(timer);
  timer = setTimeout(update, 150);
});
form.addEventListener("submit", (event) => event.preventDefault());

$("no-rider").addEventListener("click", () => {
  for (const name of ["vehicle", ...NUMBERS]) $(name).value = "";
  update();
});

async function describeModel() {
  try {
    const m = await (await fetch("/api/v1/model")).json();
    $("key-box").hidden = !m.key_required;
    $("about").textContent =
      `Typically off by ${m.average_error_minutes.toFixed(1)} minutes, measured on ` +
      `${m.test_orders.toLocaleString("en-IN")} orders the model never saw. ` +
      `Learned from Indian orders, ${m.first_day} to ${m.last_day}. Model ${m.version}.`;
  } catch {
    // The estimate itself reports connection problems.
  }
}

describeModel();
update();
