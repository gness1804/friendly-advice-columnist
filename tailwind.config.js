/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        // Custom color palette: black, white/silver, red
        primary: {
          DEFAULT: "#DC2626", // red-600
          hover: "#B91C1C", // red-700
          light: "#EF4444", // red-500
        },
        surface: {
          DEFAULT: "#0A0A0A", // near-black
          light: "#171717", // neutral-900
          lighter: "#262626", // neutral-800
        },
        text: {
          DEFAULT: "#F5F5F5", // neutral-100 (white/silver)
          muted: "#A3A3A3", // neutral-400 (silver)
          dark: "#737373", // neutral-500
        },
      },
    },
  },
  plugins: [],
};
