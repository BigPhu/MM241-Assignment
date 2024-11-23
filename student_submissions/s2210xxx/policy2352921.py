from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2352921(Policy):
    def __init__(self):
        self.patterns = []  # List of patterns (columns)
        self.num_products = 0

    def get_action(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        self.num_products = len(products)

        # if not self.patterns:
        self.patterns = self.generate_initial_patterns(products)

        dual_prices = self.solve_master_problem(self.patterns, products)
        new_pattern, reduced_cost = self.solve_subproblem(stocks, products, dual_prices)

        if reduced_cost < 0:
            self.patterns.append(new_pattern)
        else:
            action = self.select_best_action(stocks, products)
            return action

        return {"stock_idx": 0, "size": [0, 0], "position": [0, 0]}

    def generate_initial_patterns(self, products):
        """
        Generate initial patterns (e.g., one piece per pattern).
        Each pattern corresponds to a feasible cutting configuration.
        """
        num_products = len(products)
        patterns = []

        for i, product in enumerate(products):
            if product["quantity"] > 0:
                # Create a pattern that cuts one piece of this type
                pattern = [0] * num_products
                pattern[i] = 1
                patterns.append(pattern)

        # Ensure patterns is a 2D numpy array with shape (num_patterns, num_products)
        return np.array(patterns)


    def solve_master_problem(self, patterns, products):
        """
        Solve the master problem for the current set of patterns.
        """
        # Convert patterns to 2D numpy array
        patterns = np.array(patterns)
        if len(patterns.shape) != 2:
            raise ValueError("Patterns must be a 2D array where each row is a pattern.")

        # Dimensions of patterns
        num_patterns, num_products = patterns.shape

        # Ensure the number of products matches the number of columns in patterns
        if num_products != len(products):
            raise ValueError(
                f"Mismatch between number of products ({len(products)}) "
                f"and pattern columns ({num_products})."
            )

        # Objective: Minimize the total number of sheets used
        c = np.ones(num_patterns)

        # Constraints: Ensure the demand for each product is met
        A_eq = patterns.T  # Transpose patterns
        b_eq = np.array([product["quantity"] for product in products])

        # Ensure b_eq length matches A_eq rows
        if len(b_eq) != A_eq.shape[0]:
            raise ValueError(
                f"b_eq must be a 1D array with length equal to the number of rows in A_eq. "
                f"Got b_eq length {len(b_eq)} and A_eq rows {A_eq.shape[0]}."
            )

        # Solve the linear program
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")

        if result.success:
            # Compute dual prices if available
            return result.y if hasattr(result, "y") else self.compute_dual_prices(A_eq, b_eq, result.x)
        else:
            raise ValueError(f"Master problem failed to solve: {result.message}")


    def compute_dual_prices(self, A_eq, b_eq, x_solution):
        """
        Compute approximate dual variables by solving auxiliary LPs.
        """
        num_constraints = len(b_eq)
        dual_prices = []

        for i in range(num_constraints):
            # Slightly perturb the right-hand side of each constraint
            perturbed_b_eq = b_eq.copy()
            perturbed_b_eq[i] += 1e-6  # Small positive perturbation

            # Solve the perturbed LP
            result = linprog(
                np.zeros_like(x_solution), A_eq=A_eq, b_eq=perturbed_b_eq, bounds=(0, None), method="highs"
            )

            if result.success:
                # Dual price is the difference in the objective value divided by the perturbation
                dual_price = (result.fun - sum(x_solution)) / 1e-6
            else:
                dual_price = 0  # Default to zero if perturbed problem fails

            dual_prices.append(dual_price)

        return np.array(dual_prices)

    def solve_subproblem(self, stocks, products, dual_prices):
        num_products = len(products)
        best_pattern = None
        best_reduced_cost = float("inf")

        for stock in stocks:
            stock_width = np.sum(np.any(stock != -2, axis=1))
            stock_height = np.sum(np.any(stock != -2, axis=0))

            pattern = [0] * num_products
            reduced_cost = 1

            for i, product in enumerate(products):
                size = product["size"]
                width, height = size
                if width <= stock_width and height <= stock_height:
                    reduced_cost -= dual_prices[i]
                    pattern[i] += 1

            if reduced_cost < best_reduced_cost:
                best_pattern = pattern
                best_reduced_cost = reduced_cost

        return best_pattern, best_reduced_cost

    def select_best_action(self, stocks, products):
        for stock_idx, stock in enumerate(stocks):
            stock_width = np.sum(np.any(stock != -2, axis=1))
            stock_height = np.sum(np.any(stock != -2, axis=0))

            for i, product in enumerate(products):
                size = product["size"]
                quantity = product["quantity"]

                if quantity > 0:
                    width, height = size

                    for x in range(stock_width - width + 1):
                        for y in range(stock_height - height + 1):
                            if np.all(stock[x : x + width, y : y + height] == -1):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": size,
                                    "position": [x, y],
                                }

        return {"stock_idx": 0, "size": [0, 0], "position": [0, 0]}
