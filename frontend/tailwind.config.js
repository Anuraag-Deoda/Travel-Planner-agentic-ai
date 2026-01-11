/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  // Safelist dynamic classes used in the app
  safelist: [
    // Background colors
    'bg-teal-50', 'bg-teal-100', 'bg-teal-500', 'bg-teal-600',
    'bg-blue-50', 'bg-blue-100', 'bg-blue-500', 'bg-blue-600',
    'bg-indigo-50', 'bg-indigo-100', 'bg-indigo-500', 'bg-indigo-600',
    'bg-emerald-50', 'bg-emerald-100', 'bg-emerald-500', 'bg-emerald-600',
    'bg-amber-50', 'bg-amber-100', 'bg-amber-500', 'bg-amber-600',
    'bg-orange-50', 'bg-orange-100', 'bg-orange-500', 'bg-orange-600',
    'bg-green-50', 'bg-green-100', 'bg-green-500', 'bg-green-600',
    'bg-violet-50', 'bg-violet-100', 'bg-violet-500', 'bg-violet-600',
    'bg-cyan-50', 'bg-cyan-100', 'bg-cyan-500', 'bg-cyan-600',
    // Text colors
    'text-teal-500', 'text-teal-600', 'text-teal-700',
    'text-blue-500', 'text-blue-600', 'text-blue-700',
    'text-indigo-500', 'text-indigo-600', 'text-indigo-700',
    'text-emerald-500', 'text-emerald-600', 'text-emerald-700',
    'text-amber-500', 'text-amber-600', 'text-amber-700',
    'text-orange-500', 'text-orange-600', 'text-orange-700',
    'text-green-500', 'text-green-600', 'text-green-700',
    'text-violet-500', 'text-violet-600', 'text-violet-700',
    'text-cyan-500', 'text-cyan-600', 'text-cyan-700',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      boxShadow: {
        'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
      },
    },
  },
  plugins: [],
}
