import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "media",
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          light: "#fcfcfb",
          dark: "#1a1a19",
        },
        page: {
          light: "#f9f9f7",
          dark: "#0d0d0d",
        },
        ink: {
          primary: { light: "#0b0b0b", dark: "#ffffff" },
          secondary: { light: "#52514e", dark: "#c3c2b7" },
          muted: "#898781",
        },
        line: {
          hairline: { light: "#e1e0d9", dark: "#2c2c2a" },
          baseline: { light: "#c3c2b7", dark: "#383835" },
        },
        brand: {
          DEFAULT: "#2a78d6",
          dark: "#3987e5",
        },
        status: {
          good: "#0ca30c",
          warning: "#fab219",
          serious: "#ec835a",
          critical: "#d03b3b",
        },
      },
      borderRadius: {
        DEFAULT: "12px",
      },
    },
  },
  plugins: [],
};
export default config;
