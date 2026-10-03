/**
 * Minimal ES2024 augmentation.
 *
 * typescript ~5.6 has no `es2024` lib, so `tsconfig.json` compiles against
 * ES2022 and this global declaration merges the one ES2024 API the demo uses:
 * `Promise.withResolvers` (called by the api.ts / fixtures.ts delay helpers).
 * This file intentionally has no imports or exports so the interface merges
 * into the global `PromiseConstructor` instead of becoming a module.
 */
interface PromiseConstructor {
  withResolvers<T>(): {
    promise: Promise<T>;
    resolve: (value: T | PromiseLike<T>) => void;
    reject: (reason?: unknown) => void;
  };
}
