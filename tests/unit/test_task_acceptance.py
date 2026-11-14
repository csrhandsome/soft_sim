import unittest

from soft_sim.validation.task import acceptance


class TaskAcceptance(unittest.TestCase):
    def reports(self):
        return [dict(success=True, joint_control="position", official_task_manager=True), dict(valid=True),
                dict(clamp_valid=True, positive_element_volumes=True), dict(valid=True),
                dict(rest_mesh_positive=True, small_positive_threshold_explains_mass_diagnostics=True,
                     cooking_setup_error_count=0)]

    def test_success_requires_all_independent_quality_reports(self):
        reports = self.reports()
        self.assertTrue(acceptance(*reports)["accepted"])
        for index in range(1, len(reports)):
            incomplete = reports.copy()
            incomplete[index] = {}
            report = acceptance(*incomplete)
            self.assertTrue(report["task_success"])
            self.assertFalse(report["quality_valid"])
            self.assertFalse(report["accepted"])

    def test_task_failure_is_preserved_even_when_quality_passes(self):
        reports = self.reports()
        reports[0]["success"] = False
        report = acceptance(*reports)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["task_success"])
        self.assertTrue(report["quality_valid"])

    def test_diagnostic_replay_and_unresolved_cooking_errors_cannot_pass(self):
        reports = self.reports()
        reports[0]["joint_control"] = "kinematic"
        reports[-1]["cooking_setup_error_count"] = 4
        report = acceptance(*reports)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["failed_checks"], ["native_drives", "cooking_setup"])
